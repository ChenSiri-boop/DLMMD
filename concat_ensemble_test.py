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
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_recall_curve, \
    average_precision_score
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 参数
# =========================
TEST_CSV_PATH = './data/resnet/test_meta_res.csv'  # 测试集CSV路径
BEST_MODEL_PATH = "./output3/concat_5fold_best.pth"  # 保存的最佳模型
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分类变量
CLINICAL_CATEGORICAL_VARS = ['clinic_info1', 'clinic_info2', 'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info8', 'clinic_info9']
# 连续变量
CLINICAL_CONTINUOUS_VARS = ['clinic_info6', 'clinic_info10']
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2

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

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# 加载测试集
# =========================
df_test = pd.read_csv(TEST_CSV_PATH)
df_test = preprocess_clinical_data(df_test, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)

test_loader = DataLoader(
    ClinicalOnlyDataset(df_test, transform=image_transform),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# 加载模型
# =========================
input_dim = df_test.shape[1] - 4   # ⚠️ 根据训练时的 clin_input_dim
model = MultiModalFusion(clin_input_dim=input_dim, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# 测试
# =========================
T = 2.0  # softmax 温度，大于1会让概率更平滑
test_probs, test_labels = [], []
with torch.no_grad():
    for imgs, clin, labels in test_loader:
        imgs, clin = imgs.to(DEVICE), clin.to(DEVICE)
        logits = model(imgs, clin)
        # probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs = torch.softmax(logits / T, dim=1)[:, 1].cpu().numpy()
        print("logits[:10]:", logits[:10])
        print("probs[:10]:", probs[:10])
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

df = pd.DataFrame({
    "label": test_labels,  # 真实标签
    "prob": test_probs     # 预测概率
})
df.to_csv("./output3/concat/roc_test.csv", index=False)

# =========================
# 绘制 ROC 曲线
# =========================
fpr, tpr, _ = roc_curve(test_labels, test_probs)
precision, recall, thresholds = precision_recall_curve(test_labels, test_probs)
average_precision = average_precision_score(test_labels, test_probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="#616AAA", lw=3, label=f'Ultrasound (AUC = {auc:.2f})')  # 粗蓝线
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=(0, (6, 6)))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.savefig("./output3/concat/concat_test_roc", dpi=300)
plt.show()

# PRC曲线
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="#616AAA", lw=3, label=f'Ultrasound (AP = {average_precision:.2f})')  # 粗蓝线，AP是平均精度
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left')
plt.grid(False)
plt.tight_layout()
plt.savefig("./output3/concat/concat_test_prc", dpi=300)
plt.show()