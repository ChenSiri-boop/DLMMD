# -*- coding: utf-8 -*-
import os
from random import random

import matplotlib
matplotlib.use('Agg')
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.utils.tensorboard as tb
writer = tb.SummaryWriter()

# 分类变量
CLINICAL_CATEGORICAL_VARS = ['clinic_info1', 'clinic_info2', 'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info8', 'clinic_info9']
# 连续变量
CLINICAL_CONTINUOUS_VARS = ['clinic_info6', 'clinic_info10']
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2
BATCH_SIZE = 32
EPOCHS = 70
LR = 1e-3
# LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = './data/resnet/train_meta_res.csv'
best_model_path = './output3/concat_5fold_best.pth'
# =========================
# 临床 MLP 模型
# =========================
class ClinicalMLPEncoder(nn.Module):
    """只提取临床特征，不直接分类"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        self.output_dim = 32  # 输出临床特征维度

    def forward(self, x):
        return self.net(x)

# =========================
# 图像 ResNet 模型
# =========================
class ResNetEncoder(nn.Module):
    """只提取图像特征，不直接分类"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后一层fc
        self.output_dim = resnet.fc.in_features  # ResNet18 最后fc的输入维度

    def forward(self, x_img):
        feat = self.backbone(x_img).view(x_img.size(0), -1)  # [B, output_dim]
        return feat

# =========================
# 多模态融合模型
# =========================
class MultiModalFusion(nn.Module):
    def __init__(self, clin_input_dim, num_classes=2):
        super().__init__()
        # 单模态编码器
        self.clinical_encoder = ClinicalMLPEncoder(clin_input_dim)
        self.image_encoder = ResNetEncoder()

        # 融合层
        fusion_dim = self.clinical_encoder.output_dim + self.image_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_clin):
        img_feat = self.image_encoder(x_img)        # [B, resnet_dim]
        clin_feat = self.clinical_encoder(x_clin)   # [B, 32]
        feat = torch.cat([img_feat, clin_feat], dim=1)  # 拼接
        logits = self.classifier(feat)
        return logits

def preprocess_clinical_data(df, cat_vars, cont_vars,
                              missing_thresh_high=MISSING_THRESHOLD_HIGH,
                              missing_thresh_low=MISSING_THRESHOLD_LOW):
    df = df.copy()
    for var in cat_vars:
        if var not in df.columns:
            continue
        miss_ratio = df[var].isna().mean()
        if miss_ratio > missing_thresh_high:
            df.drop(columns=[var], inplace=True)
        else:
            if miss_ratio <= missing_thresh_low:
                df[var].fillna(df[var].mode().iloc[0], inplace=True)
            else:
                df[var].fillna('缺失', inplace=True)
            df[var] = LabelEncoder().fit_transform(df[var].astype(str))

    for var in cont_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df[var].fillna(df[var].mean(), inplace=True)

    scaler = StandardScaler()
    df[cont_vars] = scaler.fit_transform(df[cont_vars])
    return df

class ClinicalOnlyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.feature_cols = [col for col in dataframe.columns if col not in ['patient_id', 'exam_id', 'us_path', 'label']]
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['us_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        features = torch.tensor(row[self.feature_cols].astype(float).values, dtype=torch.float32)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, features, label


def make_weighted_sampler(labels):
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def set_seed(seed=42):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 也要设
    torch.backends.cudnn.deterministic = True  # 确保结果可复现
    torch.backends.cudnn.benchmark = False     # 关闭 benchmark 随机优化

def lr_finder(model, train_loader, optimizer, criterion, device,
              init_value=1e-7, final_value=10., beta=0.98):
    """学习率查找器：从很小的学习率开始逐渐增大，绘制loss曲线并推荐学习率区间"""
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss, best_loss = 0., 1e9
    batch_num = 0
    losses, log_lrs = [], []

    model.train()
    for imgs, clin, labels in train_loader:
        batch_num += 1
        imgs, clin, labels = imgs.to(device), clin.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, clin)
        loss = criterion(outputs, labels)

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))

        loss.backward()
        optimizer.step()

        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

        # 提前停止：loss开始爆炸
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

    # === 绘图 ===
    plt.figure(figsize=(6,4))
    plt.plot(log_lrs, losses)
    plt.xlabel("log10(Learning Rate)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.savefig('./output3/concat/lr.png')
    plt.show()

    # === 自动推荐学习率区间 ===
    losses = np.array(losses)
    log_lrs = np.array(log_lrs)
    min_idx = np.argmin(losses)
    lr_min = 10 ** log_lrs[min_idx]  # 最小loss对应的学习率
    lower = lr_min / 10
    upper = lr_min / 3
    print(f"\n>>> 学习率曲线最小值出现在 lr = {lr_min:.2e}")
    print(f">>> 建议学习率范围: {lower:.1e} ~ {upper:.1e}")


# =========================

def main():
    set_seed(42)
    # if not os.path.exists('./output3/concat'):
    #     os.makedirs('./output3/concat')
    df_all = pd.read_csv(CSV_PATH)
    df_all = preprocess_clinical_data(df_all, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)
    X = df_all.drop(columns=['label'])
    y = df_all['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_acc = 0.0
    best_auc = 0
    all_val_probs = []
    all_val_labels = []
    all_val_df = []
    all_train_df = []
    mean_fpr = np.linspace(0, 1, 100)  # 用于插值
    plt.figure(figsize=(6, 6))
    tprs = []  # 存储每一折的插值 TPR
    aucs = []  # 存储每一折 AUC

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")
        # 数据集
        train_df = df_all.iloc[train_idx]
        val_df = df_all.iloc[val_idx]
        sampler = make_weighted_sampler(train_df['label'].values)
        train_loader = DataLoader(ClinicalOnlyDataset(train_df, transform=image_transform), batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(ClinicalOnlyDataset(val_df, transform=image_transform), batch_size=BATCH_SIZE)

        # 模型
        input_dim = train_df.shape[1] - 4
        model = MultiModalFusion(clin_input_dim=input_dim, num_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # ------------------------------------
        # if fold == 1:  # 只在第一折运行一次即可
        #     print(">>> Running LR Finder ...")
        #     lr_finder(model, train_loader, optimizer, criterion, DEVICE)
        #     return
        # --------------------------------------------

        # 训练
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for imgs, clin, labels in train_loader:
                imgs, clin, labels = imgs.to(DEVICE), clin.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(imgs, clin)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Fold {fold} | Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")
            writer.add_scalar(f'Fold_{fold}/Train_Loss', epoch_loss, epoch + 1)

        # 验证
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for imgs, clin, labels in val_loader:
                imgs, clin = imgs.to(DEVICE), clin.to(DEVICE)
                logits = model(imgs, clin)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())

        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        all_val_probs.extend(val_probs)
        all_val_labels.extend(val_labels)

        # 计算 AUC
        auc = roc_auc_score(val_labels, val_probs)
        aucs.append(auc)
        print(f"Fold {fold} AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_fold = fold
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved from Fold {fold} with AUC = {auc:.4f}")

        # 绘制 ROC
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=f'Fold {fold} (AUC={auc:.3f})')

        # 计算平均 ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = roc_auc_score(all_val_labels, all_val_probs)
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC={mean_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("5-Fold Cross-Validation ROC")
    plt.legend(loc="lower right")
    # plt.savefig('./output3/concat/lr.jpg')
    plt.show()


if __name__ == "__main__":
    main()