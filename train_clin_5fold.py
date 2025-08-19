import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import interp
BATCH_SIZE = 32
EPOCHS = 70
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = './output3/clin_only_5fold.pt'
CSV_PATH = './data/resnet/train_meta_res.csv'
CLINICAL_CATEGORICAL_VARS = ['clinic_info1', 'clinic_info2', 'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info8', 'clinic_info9']
CLINICAL_CONTINUOUS_VARS = ['clinic_info6', 'clinic_info10']
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2

def make_weighted_sampler(labels):
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

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
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 也要设
    torch.backends.cudnn.deterministic = True  # 确保结果可复现
    torch.backends.cudnn.benchmark = False     # 关闭 benchmark 随机优化

if __name__ == '__main__':
    set_seed(42)
    df_all = pd.read_csv(CSV_PATH)
    df_all = preprocess_clinical_data(df_all, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)
    X = df_all.drop(columns=['label'])
    y = df_all['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_acc = 0.0
    all_val_probs = []
    all_val_labels = []
    all_val_df = []
    all_train_df = []
    mean_fpr = np.linspace(0, 1, 100)  # 用于插值
    plt.figure(figsize=(6, 6))
    tprs = []  # 存储每一折的插值 TPR
    aucs = []  # 存储每一折 AUC

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n==== Fold {fold} ====")
        train_df = df_all.iloc[train_idx]
        val_df = df_all.iloc[val_idx]

        input_dim = train_df.shape[1] - 4
        sampler = make_weighted_sampler(train_df['label'].values)
        train_loader = DataLoader(ClinicalOnlyDataset(train_df), batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(ClinicalOnlyDataset(val_df), batch_size=BATCH_SIZE)

        model = ClinicalMLPClassifier(input_dim=input_dim).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

            model.eval()
            correct, total = 0, 0
            all_probs, all_labels = [], []
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    outputs = model(features)
                    _, preds = torch.max(outputs, 1)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            fold_auc = roc_auc_score(all_labels, all_probs)
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}  Val Accuracy: {acc:.4f}")

        # 计算每折 ROC 曲线
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold_auc)
        plt.plot(fpr, tpr, lw=3, alpha=0.7, label=f'Fold {fold} (AUC = {fold_auc:.2f})')

        model.eval()
        probs_val, labels_val = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                probs_val.extend(probs.cpu().numpy())
                labels_val.extend(labels.cpu().numpy())
        all_val_probs.append(probs_val)
        all_val_labels.append(labels_val)
        print(all_val_labels)
        val_auc = roc_auc_score(labels_val, probs_val)
        print(f"Fold {fold} AUC: {val_auc:.4f}")

        val_df_with_probs = val_df.copy()
        val_df_with_probs['prob'] = probs_val
        val_df_with_probs['true_label'] = labels_val
        all_val_df.append(val_df_with_probs)

        probs_train = []
        with torch.no_grad():
            for features, labels in train_loader:
                features = features.to(DEVICE)
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                probs_train.extend(probs.cpu().numpy())

        train_df_with_probs = train_df.copy()
        train_df_with_probs['prob'] = probs_train
        all_train_df.append(train_df_with_probs)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f">>> New best Accuracy: {best_acc:.4f}, saved model to {BEST_MODEL_PATH}")

    print(f"\n5-fold CV done. Best Accuracy: {best_acc:.4f}")

    # all_df = pd.concat(all_train_df + all_val_df, axis=0).sort_values(by='exam_id').reset_index(drop=True)
    # all_df.to_csv('./output2/clin_train_val_probs.csv', index=False)
    # print("已保存所有训练集和验证集预测概率文件：./output2/clin_train_val_probs.csv")
    # 初始化变量

    # 平均 ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='black',
             label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
             lw=2.5, linestyle='--')

    # 标准差阴影
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2, label='± 1 std. dev.')

    # 图形美化
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    # 设置 x 轴（FPR）和 y 轴（TPR）的显示范围稍微超出 [0, 1]，防止图像边缘被截断
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12, fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=12, fontname='Arial')
    plt.title('5-Fold Cross-Validation ROC Curve', fontsize=12, fontname='Arial')
    plt.legend(loc='lower right', fontsize=11, prop={'family': 'Arial', 'weight': 'bold'})
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./output3/3_clin_5fold_roc.png', dpi=300)
    plt.show()

