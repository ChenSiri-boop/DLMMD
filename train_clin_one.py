import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score


BATCH_SIZE = 32
EPOCHS = 70
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = './output2/clin_only.pt'
# CLINICAL_CATEGORICAL_VARS = ['clinic_info6', 'clinic_info2', 'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info9']
# CLINICAL_CONTINUOUS_VARS = ['clinic_info1', 'clinic_info7', 'clinic_info8', 'clinic_info10']
# 分类变量
CLINICAL_CATEGORICAL_VARS = ['clinic_info1', 'clinic_info2', 'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info8', 'clinic_info9']
# 连续变量
CLINICAL_CONTINUOUS_VARS = ['clinic_info6', 'clinic_info10']
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2

# 标签平衡采样器
def make_weighted_sampler(labels):
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# 数据预处理
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

# 自定义数据集
class ClinicalOnlyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.feature_cols = [col for col in dataframe.columns if col not in ['patient_id', 'exam_id', 'us_path', 'label']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].astype(float).values, dtype=torch.float32)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return features, label

# MLP模型
class ClinicalMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    train_df = pd.read_csv('./output2/best_fold_train.csv')
    val_df = pd.read_csv('./output2/best_fold_val.csv')

    input_dim = train_df.shape[1] - 4  # 除去 patient_id, exam_id, us_path, label
    print(f"Input dim: {input_dim}")

    # DataLoader
    sampler = make_weighted_sampler(train_df['label'].values)
    train_loader = DataLoader(ClinicalOnlyDataset(train_df), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(ClinicalOnlyDataset(val_df), batch_size=BATCH_SIZE)

    # 初始化模型与优化器
    model = ClinicalMLPClassifier(input_dim=input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}  Val Accuracy: {acc:.4f}")

    # 验证集保存
    model.eval()
    all_probs_val, all_labels_val = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs_val.extend(probs.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels_val, all_probs_val)
    print(f"\nFinal Validationl AUC: {auc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f">>> 模型已保存到: {BEST_MODEL_PATH}")

    val_df_with_probs = val_df.copy()
    val_df_with_probs['prob'] = all_probs_val
    val_df_with_probs['true_label'] = all_labels_val

    # 生成训练集概率
    all_probs_train, all_labels_train = [], []
    with torch.no_grad():
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs_train.extend(probs.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())

    train_df_with_probs = train_df.copy()
    train_df_with_probs['prob'] = all_probs_train
    train_df_with_probs['true_label'] = all_labels_train

    # ========= 合并并保存 ==========
    combined_df = pd.concat([train_df_with_probs, val_df_with_probs], axis=0)
    combined_df_sorted = combined_df.sort_values(by='exam_id').reset_index(drop=True)

    # 保存为 CSV
    combined_df_sorted.to_csv('./output2/clin_train_val_probs.csv', index=False)


