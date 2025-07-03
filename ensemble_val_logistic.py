import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# 加载文件
img_df = pd.read_csv('./output2/img_train_val_probs.csv')
clinical_df = pd.read_csv('./output2/clin_train_val_probs.csv')

# 提取为 numpy 数组
img_probs = img_df['prob'].values
clinical_probs = clinical_df['prob'].values
labels = img_df['label'].values

# 检查维度一致
assert len(img_probs) == len(clinical_probs) == len(labels), "维度不一致"

# ========================
# 使用 Logistic Regression 学习融合权重
# ========================
X = np.stack([img_probs, clinical_probs], axis=1)  # shape: [N, 2]
y = labels

# 使用无正则项的逻辑回归拟合（惩罚设置为 none 或 C 很大）
model = LogisticRegression(penalty='none', solver='lbfgs')
model.fit(X, y)

# 获取融合概率
fused_probs = model.predict_proba(X)[:, 1]
fused_preds = (fused_probs >= 0.5).astype(int)

# 评估指标
acc = accuracy_score(y, fused_preds)
auc = roc_auc_score(y, fused_probs)

# Logistic Regression learned weights:  [[9.71960976 2.07698987]]
# Logistic Regression bias:  [-7.18312892]
print("Logistic Regression learned weights: ", model.coef_)
print("Logistic Regression bias: ", model.intercept_)
print("融合后 Accuracy:", f"{acc:.4f}")
print("融合后 AUC:", f"{auc:.4f}")

# ========================
# 绘制 ROC 曲线
# ========================
# fpr, tpr, _ = roc_curve(y, fused_probs)
#
# plt.figure()
# plt.plot(fpr, tpr, label=f'Logistic Fusion ROC (AUC = {auc:.4f})', color='darkred', lw=2)
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve (Logistic Regression Fusion)')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.savefig('./output/val_ensemble_logistic_roc.png', dpi=300, transparent=True)
# plt.show()
