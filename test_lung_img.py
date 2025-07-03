import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import pickle


# 配置
TEST_CSV_PATH = './data/resnet/test_lung_meta_res.csv'
BEST_MODEL_PATH = './output2/lung_img.pt'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

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

# 测试集Dataset
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

test_df = pd.read_csv(TEST_CSV_PATH)
test_dataset = ImageOnlyDataset(test_df, transform=image_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载模型
model = ResNetOnlyClassifier(num_classes=2)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

# 性能评估
acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")

# 保存前按 exam_id 排序
test_df['prob'] = all_probs
test_df_sorted = test_df.sort_values(by='exam_id').reset_index(drop=True)
#
#  保存预测概率
test_df_sorted[['exam_id', 'label', 'prob']].to_csv('./output2/lung_image_test_probs.csv', index=False)
print("图像预测概率已保存。")

# # 计算 ROC 曲线数据
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
average_precision = average_precision_score(all_labels, all_probs)

with open("./output2/lung_img_roc.pkl", "wb") as f:
    pickle.dump((all_labels, all_probs), f)

# 绘制 ROC 曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="#616AAA", lw=3, label=f'Ultrasound (AUC = {auc:.2f})')  # 粗蓝线
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=(0, (6, 6)))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.savefig("./output2/lung_img_roc.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="#616AAA", lw=3, label=f'Ultrasound (AP = {average_precision:.2f})')  # 粗蓝线，AP是平均精度
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left')
plt.grid(False)
plt.tight_layout()
plt.savefig("./output2/lung_img_prc.png", dpi=300)
plt.show()