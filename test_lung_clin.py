import pickle

import torch
import matplotlib
matplotlib.use('Agg')
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

# 配置
TEST_CSV = './data/resnet/test_lung_meta_res.csv'
BEST_MODEL_PATH = './output2/lung_clin.pt'
BATCH_SIZE = 32
CLINICAL_CATEGORICAL_VARS = ['clinic_info1',  'clinic_info3', 'clinic_info4', 'clinic_info5', 'clinic_info7', 'clinic_info9', 'clinic_info6', 'clinic_info10']
CLINICAL_CONTINUOUS_VARS = ['clinic_info2',  'clinic_info8']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MISSING_THRESHOLD_HIGH = 0.3
MISSING_THRESHOLD_LOW = 0.2

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

# 测试集数据集类
class ClinicalOnlyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.features = [col for col in self.df.columns if col not in ['patient_id', 'exam_id', 'us_path', 'label']]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        print("Row[self.features]:", row[self.features])
        print("Values:", row[self.features].values)
        print("Dtypes:", row[self.features].dtypes)
        # x = torch.tensor(row[self.features].values, dtype=torch.float32)
        x = torch.tensor(row[self.features].astype(float).values, dtype=torch.float32)
        y = torch.tensor(int(row['label']), dtype=torch.long)
        return x, y

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

# 加载数据
df_test = pd.read_csv(TEST_CSV)
df_test = preprocess_clinical_data(df_test, CLINICAL_CATEGORICAL_VARS, CLINICAL_CONTINUOUS_VARS)

# 构建 DataLoader
test_dataset = ClinicalOnlyDataset(df_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 模型加载
clinical_dim = len(df_test.columns) - 4  # 除去 patient_id, exam_id, label
model = ClinicalMLPClassifier(input_dim=clinical_dim).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

# 评估
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 计算指标
acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")

# # 保存前按 exam_id 排序，确保顺序一致
df_test['prob'] = all_probs
df_test['label'] = all_labels  # 再次写入标签确保一致
#
# # 按 exam_id 排序并保存
df_test_sorted = df_test.sort_values(by='exam_id').reset_index(drop=True)

# 保存预测概率和标签

df_test_sorted[['exam_id', 'label', 'prob']].to_csv('./output2/lung_clin_test_probs.csv', index=False)
# print("临床预测概率、标签和 exam_id 已保存。")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
average_precision = average_precision_score(all_labels, all_probs)

with open("./output2/lung_clin_roc.pkl", "wb") as f:
    pickle.dump((all_labels, all_probs), f)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="#616AAA", lw=3, label=f'Clinical (AUC = {auc:.2f})')  # 粗蓝线
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=(0, (6, 6)))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.savefig('./output2/lung_clin_test_roc.png', dpi=300)  # 保存图片
plt.show()



plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="#616AAA", lw=3, label=f'Clinical (AP = {average_precision:.2f})')  # 粗蓝线，AP是平均精度
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left')
plt.grid(False)
plt.tight_layout()
plt.savefig("./output2/lung_clin_prc.png", dpi=300)
plt.show()