import os
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import WeightedRandomSampler

def make_weighted_sampler(labels):
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# 配置
CSV_PATH = './data/resnet/train_meta_res.csv'
BATCH_SIZE = 32
EPOCHS = 70
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BEST_MODEL_PATH = './output/only_img_one.pt'
# 保存每折模型前缀
FOLD_MODEL_PREFIX = './resnet_fold'
# 分类变量
CLINICAL_CATEGORICAL_VARS = ['clinic_info1', 'clinic_info2', 'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info8', 'clinic_info9']
# 连续变量
CLINICAL_CONTINUOUS_VARS = ['clinic_info6', 'clinic_info10']
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2
OUTPUT_CSV_PATH = './output3/ensenmble_img.csv'
# 数据增强 + 标准化
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据预处理
def preprocess_clinical_data(df, cat_vars, cont_vars,
                              missing_thresh_high=MISSING_THRESHOLD_HIGH,
                              missing_thresh_low=MISSING_THRESHOLD_LOW):
    df = df.copy()
    # 分类变量处理
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
    # 连续变量处理
    for var in cont_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df[var].fillna(df[var].mean(), inplace=True)
    # 标准化连续变量
    scaler = StandardScaler()
    df[cont_vars] = scaler.fit_transform(df[cont_vars])
    return df

class ImageOnlyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['us_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        ids = row['exam_id']
        return img, label, ids


class ResNetOnlyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
        self.resnet_out_dim = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(self.resnet_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img):
        x = self.backbone(x_img).view(x_img.size(0), -1)
        return self.classifier(x)


# 不使用5折交叉验证训练，固定划分训练集和验证集
if __name__ == '__main__':
    # 读取数据
    df_all = pd.read_csv(CSV_PATH)
    y = df_all['label'].values

    # 划分训练/验证
    train_df, val_df = train_test_split(df_all, test_size=0.2, stratify=y, random_state=42)

    # 采样器
    sampler = make_weighted_sampler(train_df['label'].values)

    # DataLoader
    train_loader = DataLoader(ImageOnlyDataset(train_df, transform=image_transform),
                              batch_size=BATCH_SIZE,
                              sampler=sampler)
    val_loader = DataLoader(ImageOnlyDataset(val_df, transform=image_transform),
                            batch_size=BATCH_SIZE)

    # 模型、损失、优化器
    model = ResNetOnlyClassifier(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    best_train_probs, best_train_labels, best_train_ids = [], [], []
    best_val_probs, best_val_labels, best_val_ids = [], [], []

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels, ids in DataLoader(ImageOnlyDataset(train_df, transform=image_transform),
                                            batch_size=BATCH_SIZE,
                                            sampler=sampler):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        correct, total = 0, 0
        all_probs, all_labels, all_ids = [], [], []
        with torch.no_grad():
            for imgs, labels, ids in DataLoader(ImageOnlyDataset(val_df, transform=image_transform),
                                                batch_size=BATCH_SIZE):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(ids)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        auc = roc_auc_score(all_labels, all_probs)
        print(f"[Epoch {epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}  Val Acc: {acc:.4f}  AUC: {auc:.4f}")

        # 保存最佳模型和对应概率
        if acc > best_acc:
            best_acc = acc
            # 保存训练集结果
            best_train_probs, best_train_labels, best_train_ids = [], [], []
            with torch.no_grad():
                for imgs, labels, ids in DataLoader(ImageOnlyDataset(train_df, transform=image_transform),
                                                    batch_size=BATCH_SIZE):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    best_train_probs.extend(probs.cpu().numpy())
                    best_train_labels.extend(labels.cpu().numpy())
                    best_train_ids.extend(ids)

            # 保存验证集结果
            best_val_probs, best_val_labels, best_val_ids = all_probs, all_labels, all_ids

    # === 合并训练集和验证集结果 ===
    train_results = pd.DataFrame({
        "exam_id": best_train_ids,
        "prob": best_train_probs,
        "label_true": best_train_labels
    })
    val_results = pd.DataFrame({
        "exam_id": best_val_ids,
        "prob": best_val_probs,
        "label_true": best_val_labels
    })

    all_results = pd.concat([train_results, val_results], axis=0)
    all_results = all_results.sort_values(by="exam_id").reset_index(drop=True)

    all_results.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"训练集+验证集预测结果已保存至 {OUTPUT_CSV_PATH}")

