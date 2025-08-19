import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
from ensemble_val_logistic import w_img, w_clinical, bias
# 加载文件

# 加载文件
img_df = pd.read_csv('./output3/3_img_roc_test.csv')
clinical_df = pd.read_csv('./output3/3_clin_roc_test.csv')
# 提取为 numpy 数组
img_probs = img_df['prob'].values
clinical_probs = clinical_df['prob'].values
labels = img_df['label'].values

# 检查维度一致
assert len(img_probs) == len(clinical_probs) == len(labels), "维度不一致"

# # # 硬编码权重和截距
# 0.91
w_img = 0.30
w_clinical = 0.70
#
# w_img = 0.7
# w_clinical = 0.3

# w_img = 0.66
# w_clinical = 0.34

# sigmoid 函数定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 计算融合后的概率
logits = w_img * img_probs + w_clinical * clinical_probs
fused_probs = logits
# fused_probs = sigmoid(logits)

# 基于0.5阈值计算预测类别
# fused_preds = (fused_probs >= 0.5).astype(int)
fused_preds = (fused_probs >= 0.5).astype(int)

# 计算评估指标
acc = accuracy_score(labels, fused_preds)
auc = roc_auc_score(labels, fused_probs)

print(f"融合后 Accuracy: {acc:.4f}")
print(f"融合后 AUC: {auc:.4f}")
print(f"权重系数: img_probs = {w_img:.4f}, clinical_probs = {w_clinical:.4f}")


# 合并保存为一个 CSV
output_df = pd.DataFrame({
    # 'exam_id': exam_ids,
    'label': labels,
    'prob': fused_probs
})

# # 保存为 CSV 文件
output_df.to_csv('./output3/test_probs_labels_ensamble.csv', index=False)
print("保存完成: ./output3/test_probs_labels_ensamble_bianli.csv")

# 计算 ROC 曲线数据
fpr, tpr, _ = roc_curve(labels, fused_probs)

# 绘制 ROC 曲线并显示权重信息
plt.figure(figsize=(6, 6))
text_str = f'Weights:\nImg: {w_img:.2f}\nClinical: {w_clinical:.2f}\nIntercept: {bias:.2f}'
plt.text(0.6, 0.2, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

plt.plot(fpr, tpr, label=f'Logistic Fusion ROC (AUC = {auc:.4f})', color='darkred', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Logistic Regression Fusion)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('./output3/3_ensemble_test_roc.png', dpi=300)
plt.show()

# ======== 添加 PRC 曲线绘制 ========
precision, recall, _ = precision_recall_curve(labels, fused_probs)
ap = average_precision_score(labels, fused_probs)

plt.figure(figsize=(6, 6))
text_str = f'Weights:\nImg: {w_img:.2f}\nClinical: {w_clinical:.2f}\nIntercept: {bias:.2f}'
plt.text(0.6, 0.2, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

plt.plot(recall, precision, label=f'Logistic Fusion PRC (AP = {ap:.4f})', color='darkblue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Logistic Regression Fusion)')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./output/3_ensemble_test_prc.png', dpi=300)
plt.show()