import os
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

def make_weighted_sampler(labels):
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# 配置
CSV_PATH = './data/resnet/train_lung_meta_res.csv'
BATCH_SIZE = 32
EPOCHS = 70
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = './output2/lung_img.pt'
# 分类变量

CLINICAL_CATEGORICAL_VARS = ['clinic_info1',  'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info9', 'clinic_info6', 'clinic_info10']
CLINICAL_CONTINUOUS_VARS = ['clinic_info2',  'clinic_info8']
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2

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
        return img, label


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

# 训练一个epoch
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, val_loader):
    model.eval()
    all_probs, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    auc_score = roc_auc_score(all_labels, all_probs)
    return acc, auc_score

def predict_probs(model, dataloader):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            probs_list.extend(probs.cpu().numpy())
    return probs_list

def save_best_fold(model, train_df, val_df, all_probs_val, best_acc):
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f">>> New best Accuracy: {best_acc:.4f}, saved model to {BEST_MODEL_PATH}")

    # 保存训练集和验证集的预测概率
    train_loader = DataLoader(ImageOnlyDataset(train_df, transform=image_transform),
                              batch_size=BATCH_SIZE)
    train_probs = predict_probs(model, train_loader)

    train_df_with_probs = train_df.copy()
    train_df_with_probs['prob'] = train_probs
    train_df_sorted = train_df_with_probs.sort_values(by='exam_id').reset_index(drop=True)

    val_df_with_probs = val_df.copy()
    val_df_with_probs['prob'] = all_probs_val
    val_df_sorted = val_df_with_probs.sort_values(by='exam_id').reset_index(drop=True)

    combined_df = pd.concat([train_df_sorted, val_df_sorted], axis=0).reset_index(drop=True)
    combined_df.to_csv('./output2/lung_img_train_val_probs.csv', index=False)
    print("最优折训练集和验证集预测概率已保存至 './output2/lung_img_train_val_probs.csv'")

    # 删除 'prob' 列，确保只保留原始特征
    train_df_no_prob = train_df_sorted.drop(columns=['prob'])
    val_df_no_prob = val_df_sorted.drop(columns=['prob'])

    # 再进行临床特征预处理
    train_df_processed = preprocess_clinical_data(train_df_no_prob, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)
    val_df_processed = preprocess_clinical_data(val_df_no_prob, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)

    # 保存
    train_df_processed.to_csv('./output2/lung_best_fold_train.csv', index=False)
    val_df_processed.to_csv('./output2/lung_best_fold_val.csv', index=False)
    print("已保存最佳折训练集和验证集数据（已去除 prob 列，按 exam_id 排序）")

# 5折交叉验证训练，保存每折模型并保留最佳准确率模型
if __name__ == '__main__':
    # 读取并预处理全部数据
    df_all = pd.read_csv(CSV_PATH)
    X = df_all.drop(columns=['label'])
    y = df_all['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_acc = 0.0  # 保存5折中表现最好的模型准确率
    tprs = []  # 存储每一折的插值 TPR
    aucs = []  # 存储每一折 AUC
    mean_fpr = np.linspace(0, 1, 100)  # 用于插值
    plt.figure(figsize=(6, 6))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n==== Fold {fold} ====")
        train_df = df_all.iloc[train_idx]
        val_df = df_all.iloc[val_idx]
        # 加载数据
        sampler = make_weighted_sampler(train_df['label'].values)
        train_loader = DataLoader(ImageOnlyDataset(train_df, transform=image_transform),
                                  batch_size=BATCH_SIZE,
                                  sampler=sampler)
        val_loader = DataLoader(ImageOnlyDataset(val_df, transform=image_transform),
                                batch_size=BATCH_SIZE)

        # 模型、损失和优化器
        model = ResNetOnlyClassifier(num_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        # 训练
        for epoch in range(EPOCHS):
            avg_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            acc, auc_score = validate(model, val_loader)
            print(f"[Epoch {epoch+1}/{EPOCHS}] Training Loss: {avg_loss:.4f}  Validation Accuracy: {acc:.4f}")

        # 计算并输出折验证准确率
        fold_correct, fold_total = 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE),  labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                fold_correct += (preds == labels).sum().item()
                fold_total += labels.size(0)
        fold_acc = fold_correct / fold_total
        fold_auc = roc_auc_score(all_labels, all_probs)
        print(f"Fold {fold} Accuracy: {fold_acc:.4f}")
        print(f"Fold {fold} AUC: {fold_auc:.4f}")

        # 计算每折 ROC 曲线
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold_auc)
        plt.plot(fpr, tpr, lw=3, alpha=0.7, label=f'Fold {fold} (AUC = {fold_auc:.2f})')

        # 保存最优模型（基于准确率）保存最优折预测概率、训练集和验证集
        if fold_acc > best_acc:
            best_acc = fold_acc
            save_best_fold(model, train_df, val_df, all_probs, best_acc)

    print(f"\n5-fold CV done. Best Accuracy: {best_acc:.4f}")

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
    plt.savefig('./output2/lung_img_5fold_roc.png', dpi=300)
    plt.show()

