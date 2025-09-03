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
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.utils.tensorboard as tb
writer = tb.SummaryWriter()

# =========================
# 配置
# =========================
# bone 任务示例
CLINICAL_CATEGORICAL_VARS = ['clinic_info6', 'clinic_info2', 'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info9']
CLINICAL_CONTINUOUS_VARS = ['clinic_info1', 'clinic_info7', 'clinic_info8', 'clinic_info10']

MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LR = 6.3e-04
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据与输出路径
CSV_PATH = './data/resnet/train_bone_meta_res.csv'
best_model_path = './output3/bone/LORA/concat_5fold_best_adapter.pth'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

# =========================
# 临床 MLP 编码器（仅特征提取）
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
# Adapter 模块（参数高效）
# =========================
class Adapter(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        bottleneck = max(1, dim // reduction)
        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim)
        )

    def forward(self, x):
        # 残差连接：输出 = 输入 + 适配器调整
        return x + self.adapter(x)

# =========================
# 图像编码器：ResNet18 + Adapter（冻结主干，只训 Adapter）
# =========================
class ResNetEncoderWithAdapter(nn.Module):
    """只提取图像特征，不直接分类；主干冻结，仅训练 Adapter"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后一层fc
        self.output_dim = resnet.fc.in_features  # 512 for resnet18

        # 冻结主干
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 在输出特征上加 Adapter（可按需加多层，这里简单放一层在尾部）
        self.adapter = Adapter(self.output_dim, reduction=16)

    def forward(self, x_img):
        feat = self.backbone(x_img).view(x_img.size(0), -1)  # [B, output_dim]
        feat = self.adapter(feat)  # 仅 Adapter 参与训练
        return feat

# =========================
# 多模态融合模型
# =========================
class MultiModalFusion(nn.Module):
    def __init__(self, clin_input_dim, num_classes=2):
        super().__init__()
        # 单模态编码器
        self.clinical_encoder = ClinicalMLPEncoder(clin_input_dim)
        # self.image_encoder = ResNetEncoderWithAdapter()
        self.image_encoder = ResNetEncoderWithLoRA()

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

# =========================
# 预处理：临床特征
# =========================
def preprocess_clinical_data(df, cat_vars, cont_vars,
                              missing_thresh_high=MISSING_THRESHOLD_HIGH,
                              missing_thresh_low=MISSING_THRESHOLD_LOW):
    df = df.copy()
    # 分类变量：高缺失直接丢；中等缺失填“缺失”；低缺失填众数
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

    # 连续变量：转数值 + 用均值填充 + 标准化
    for var in cont_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df[var].fillna(df[var].mean(), inplace=True)

    scaler = StandardScaler()
    kept_cont = [v for v in cont_vars if v in df.columns]
    if kept_cont:
        df[kept_cont] = scaler.fit_transform(df[kept_cont])
    return df

# =========================
# 数据集
# =========================
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
    labels_t = torch.tensor(labels)
    class_counts = torch.bincount(labels_t)
    class_weights = 1. / class_counts.float()
    weights = class_weights[labels_t]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# 可选：学习率查找器（保留原逻辑）
# =========================
def lr_finder(model, train_loader, optimizer, criterion, device,
              init_value=1e-7, final_value=10., beta=0.98):
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / max(1, num))
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
        losses.append(smoothed_loss); log_lrs.append(np.log10(lr))
        loss.backward(); optimizer.step()
        lr *= mult; optimizer.param_groups[0]['lr'] = lr
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

    plt.figure(figsize=(6,4))
    plt.plot(log_lrs, losses)
    plt.xlabel("log10(Learning Rate)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    os.makedirs('./output3/concat', exist_ok=True)
    plt.savefig('./output3/concat/lr.png')
    plt.close()

    losses = np.array(losses); log_lrs = np.array(log_lrs)
    min_idx = np.argmin(losses)
    lr_min = 10 ** log_lrs[min_idx]
    lower = lr_min / 10; upper = lr_min / 3
    print(f"\n>>> 学习率曲线最小值出现在 lr = {lr_min:.2e}")
    print(f">>> 建议学习率范围: {lower:.1e} ~ {upper:.1e}")


# =========================
# CBAM 模块
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.sharedMLP = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3))  # [B, C]
        max_out, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)  # [B, C]
        out = self.sharedMLP(avg_out) + self.sharedMLP(max_out)
        return self.sigmoid(out).unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# =========================
# 图像编码器：ResNet18 + Adapter + CBAM
# =========================
class ResNetEncoderWithAdapterCBAM(nn.Module):
    """ResNet 提取特征 + CBAM 注意力 + Adapter"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 保留到最后一层 conv
        self.output_dim = resnet.fc.in_features  # 512 for resnet18

        # CBAM 加在最后卷积输出上
        self.cbam = CBAM(self.output_dim)

        # 冻结主干
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 全局池化 + Adapter
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.adapter = Adapter(self.output_dim, reduction=16)

    def forward(self, x_img):
        feat_map = self.backbone(x_img)  # [B, 512, H, W]
        feat_map = self.cbam(feat_map)   # CBAM 注意力
        feat = self.avgpool(feat_map).view(x_img.size(0), -1)  # [B, 512]
        feat = self.adapter(feat)  # Adapter
        return feat

# =========================
# LoRA 模块
# =========================
class LoRA(nn.Module):
    def __init__(self, dim, rank=8, alpha=1.0):
        """
        dim: 输入/输出维度 (与主干特征相同, 如 512)
        rank: LoRA 低秩分解维度
        alpha: 缩放因子
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # 低秩分解参数
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)

        # 初始化：A 正常初始化，B 初始化为 0（保证初始等价于恒等映射）
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)

        # 冻结主干时，只训练 LoRA 参数
        for p in self.A.parameters():
            p.requires_grad = True
        for p in self.B.parameters():
            p.requires_grad = True

    def forward(self, x):
        return x + self.alpha * self.B(self.A(x))

class ResNetEncoderWithLoRA(nn.Module):
    """ResNet 提取特征 + CBAM 注意力 + LoRA"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 保留到最后一层 conv
        self.output_dim = resnet.fc.in_features  # 512 for resnet18

        # CBAM 注意力
        self.cbam = CBAM(self.output_dim)

        # 冻结主干
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 全局池化 + LoRA
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lora = LoRA(self.output_dim, rank=8, alpha=1.0)

    def forward(self, x_img):
        feat_map = self.backbone(x_img)  # [B, 512, H, W]
        feat_map = self.cbam(feat_map)   # CBAM 注意力
        feat = self.avgpool(feat_map).view(x_img.size(0), -1)  # [B, 512]
        feat = self.lora(feat)  # LoRA 替代 Adapter
        return feat

# =========================
# 主函数：5 折训练 + ROC 绘制
# =========================
def main():
    set_seed(42)
    df_all = pd.read_csv(CSV_PATH)
    df_all = preprocess_clinical_data(df_all, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)
    X = df_all.drop(columns=['label'])
    y = df_all['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_auc = 0.0
    all_val_probs = []
    all_val_labels = []
    mean_fpr = np.linspace(0, 1, 100)  # 用于插值
    plt.figure(figsize=(6, 6))
    tprs = []  # 存储每一折的插值 TPR
    aucs = []  # 存储每一折 AUC

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")
        # 数据集
        train_df = df_all.iloc[train_idx].reset_index(drop=True)
        val_df = df_all.iloc[val_idx].reset_index(drop=True)

        # 计算临床输入维度（更稳妥：按特征列数来推断）
        feature_cols = [c for c in train_df.columns if c not in ['patient_id', 'exam_id', 'us_path', 'label']]
        input_dim = len(feature_cols)

        sampler = make_weighted_sampler(train_df['label'].values)
        train_loader = DataLoader(ClinicalOnlyDataset(train_df, transform=image_transform),
                                  batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(ClinicalOnlyDataset(val_df, transform=image_transform),
                                batch_size=BATCH_SIZE)

        # 模型（ResNet + Adapter + 临床 MLP）
        model = MultiModalFusion(clin_input_dim=input_dim, num_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        # 只优化 requires_grad=True 的参数（即 Adapter + 临床 MLP + 分类头）
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

        # ------------------------------------
        # 可选：第一折做一次 LR Finder
        # if fold == 1:
        #     print(">>> Running LR Finder ...")
        #     lr_finder(model, train_loader, optimizer, criterion, DEVICE)
        #     return
        # ------------------------------------

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

            epoch_loss = running_loss / max(1, len(train_loader))
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
        fold_auc = roc_auc_score(val_labels, val_probs)
        aucs.append(fold_auc)
        print(f"Fold {fold} AUC = {fold_auc:.4f}")

        # 保存最优
        if fold_auc > best_auc:
            best_auc = fold_auc
            best_fold = fold
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved from Fold {fold} with AUC = {fold_auc:.4f}")

        # ROC 曲线（每折）
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=f'Fold {fold} (AUC={fold_auc:.3f})')

    # 平均 ROC
    mean_tpr = np.mean(tprs, axis=0)
    overall_auc = roc_auc_score(all_val_labels, all_val_probs)
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC={overall_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("5-Fold Cross-Validation ROC (ResNet + Adapter + BCAM)")
    plt.legend(loc="lower right")

    # os.makedirs('./output3/bone', exist_ok=True)
    fig_path = './output3/bone/LORA/roc_5fold_adapter.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📈 每折 AUC: {[round(a,4) for a in aucs]}")
    print(f"🏆 平均/总体 AUC: {overall_auc:.4f}")
    print(f"💾 最佳折：Fold {best_fold} | 最佳模型路径: {best_model_path}")
    print(f"🖼 ROC 图保存至: {fig_path}")

if __name__ == "__main__":
    main()
