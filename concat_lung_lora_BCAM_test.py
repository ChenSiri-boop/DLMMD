# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_curve, average_precision_score

# =========================
# 参数
# =========================
TEST_CSV_PATH = './data/resnet/test_lung_meta_res.csv'   # 测试集CSV路径
BEST_MODEL_PATH = './output3/lung/LORA/concat_5fold_best_adapter.pth'  # 训练保存的最佳模型
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分类变量 & 连续变量
CLINICAL_CATEGORICAL_VARS = ['clinic_info1',  'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info9', 'clinic_info6', 'clinic_info10']
CLINICAL_CONTINUOUS_VARS = ['clinic_info2',  'clinic_info8']
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2

# =========================
# 数据预处理
# =========================
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
    kept_cont = [v for v in cont_vars if v in df.columns]
    if kept_cont:
        df[kept_cont] = scaler.fit_transform(df[kept_cont])
    return df

# =========================
# Dataset
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

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# 模型结构（与训练保持一致）
# =========================
class ClinicalMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        self.output_dim = 32

    def forward(self, x):
        return self.net(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.sharedMLP = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3))
        max_out, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)
        out = self.sharedMLP(avg_out) + self.sharedMLP(max_out)
        return self.sigmoid(out).unsqueeze(2).unsqueeze(3)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
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
        super().__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class LoRA(nn.Module):
    def __init__(self, dim, rank=8, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return x + self.alpha * self.B(self.A(x))

class ResNetEncoderWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.output_dim = resnet.fc.in_features
        self.cbam = CBAM(self.output_dim)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lora = LoRA(self.output_dim, rank=8, alpha=1.0)
    def forward(self, x_img):
        feat_map = self.backbone(x_img)
        feat_map = self.cbam(feat_map)
        feat = self.avgpool(feat_map).view(x_img.size(0), -1)
        feat = self.lora(feat)
        return feat

class MultiModalFusion(nn.Module):
    def __init__(self, clin_input_dim, num_classes=2):
        super().__init__()
        self.clinical_encoder = ClinicalMLPEncoder(clin_input_dim)
        self.image_encoder = ResNetEncoderWithLoRA()
        fusion_dim = self.clinical_encoder.output_dim + self.image_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x_img, x_clin):
        img_feat = self.image_encoder(x_img)
        clin_feat = self.clinical_encoder(x_clin)
        feat = torch.cat([img_feat, clin_feat], dim=1)
        logits = self.classifier(feat)
        return logits

# 计算置信区间
def bootstrap_ci(y_true, y_score, metric_func, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    stats = []
    n = len(y_true)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, n)  # 有放回采样
        if len(np.unique(y_true[indices])) < 2:
            continue  # 跳过只有单一类别的采样
        stat = metric_func(y_true[indices], y_score[indices])
        stats.append(stat)
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return lower, upper
# =========================
# 加载数据
# =========================
df_test = pd.read_csv(TEST_CSV_PATH)
df_test = preprocess_clinical_data(df_test, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)
test_loader = DataLoader(ClinicalOnlyDataset(df_test, transform=image_transform),
                         batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 加载模型
# =========================
input_dim = df_test.shape[1] - 4  # 减去 patient_id, exam_id, us_path, label
model = MultiModalFusion(clin_input_dim=input_dim, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# 测试推理
# =========================
test_probs, test_labels = [], []
with torch.no_grad():
    for imgs, clin, labels in test_loader:
        imgs, clin = imgs.to(DEVICE), clin.to(DEVICE)
        logits = model(imgs, clin)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        test_probs.extend(probs)
        test_labels.extend(labels.numpy())

test_probs = np.array(test_probs)
test_labels = np.array(test_labels)

# =========================
# 计算指标
# =========================
auc = roc_auc_score(test_labels, test_probs)
preds = (test_probs > 0.5).astype(int)
acc = accuracy_score(test_labels, preds)
print(f"Test AUC = {auc:.4f}")
print(f"Test ACC = {acc:.4f}")

# =========================
# 绘制曲线
# =========================
fpr, tpr, _ = roc_curve(test_labels, test_probs)
precision, recall, thresholds = precision_recall_curve(test_labels, test_probs)
average_precision = average_precision_score(test_labels, test_probs)
auc_ci = bootstrap_ci(test_labels, test_probs, roc_auc_score)
ap_ci = bootstrap_ci(test_labels, test_probs, average_precision_score)

# =========================
# 绘制 ROC 曲线
# =========================
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="#616AAA", lw=3,
         label=f'Fusion (AUC = {auc:.2f} [{auc_ci[0]:.2f}, {auc_ci[1]:.2f}])')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=(0, (6, 6)))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.savefig("./output3/lung/LORA/concat_test_roc_CI", dpi=300)
plt.show()

# 保存预测结果
pd.DataFrame({
    "label": test_labels,
    "prob": test_probs
}).to_csv("./output3/lung/LORA/test_predictions.csv", index=False)

# PRC曲线
plt.figure(figsize=(6, 6))
# plt.plot(recall, precision, color="#616AAA", lw=3, label=f'Ultrasound (AP = {average_precision:.2f})')  # 粗蓝线，AP是平均精度
plt.plot(recall, precision, color="#616AAA", lw=3,
         label=f'Fusion (AP = {average_precision:.2f} [{ap_ci[0]:.2f}, {ap_ci[1]:.2f}])')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left')
plt.grid(False)
plt.tight_layout()
plt.savefig("./output3/lung/LORA/concat_test_prc_CI", dpi=300)
plt.show()