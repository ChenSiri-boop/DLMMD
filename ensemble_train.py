import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# 加载文件
img_df = pd.read_csv('./output3/ensenmble_img.csv')
clinical_df = pd.read_csv('./output3/ensenmble_clin.csv')

# 提取为 numpy 数组
img_probs = img_df['prob'].values
clinical_probs = clinical_df['prob'].values
labels = img_df['label_true'].values

# 检查维度一致
assert len(img_probs) == len(clinical_probs) == len(labels), "维度不一致"

# ========================
# 遍历不同权重寻找最佳 AUC
# ========================
best_auc = 0.0
best_alpha = 0.0
best_fused_probs = None

alphas = np.linspace(0, 1, 101)  # alpha 从 0 到 1，每隔 0.01 步长

for alpha in alphas:
    beta = 1 - alpha
    fused_probs = alpha * img_probs + beta * clinical_probs
    auc = roc_auc_score(labels, fused_probs)

    if auc > best_auc:
        best_auc = auc
        best_alpha = alpha
        best_fused_probs = fused_probs

# 最佳预测和评估
best_beta = 1 - best_alpha
fused_preds = (best_fused_probs >= 0.5).astype(int)
acc = accuracy_score(labels, fused_preds)

print(f'最佳权重 α = {best_alpha:.2f}, β = {best_beta:.2f}')
print(f'融合后 Accuracy: {acc:.4f}')
print(f'融合后 AUC: {best_auc:.4f}')

# ========================
# 绘制 ROC 曲线
# ========================
# fpr, tpr, _ = roc_curve(labels, best_fused_probs)
#
# plt.figure()
# plt.plot(fpr, tpr, label=f'Fused ROC (AUC = {best_auc:.4f})', color='darkred', lw=2)
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'Optimal Fusion ROC Curve (α={best_alpha:.2f}, β={best_beta:.2f})')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.savefig('./output/ensemble_adjust_weight_roc.png', dpi=300)
# plt.show()
