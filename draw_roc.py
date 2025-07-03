import matplotlib
matplotlib.use('Agg')
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_and_compute(path):
    with open(path, "rb") as f:
        all_lables, all_probs = pickle.load(f)
    fpr, tpr, _ = roc_curve(all_lables, all_probs)
    auc_val = auc(fpr, tpr)
    return fpr, tpr, auc_val

# 加载三个模型的结果
fpr1, tpr1, auc1 = load_and_compute("./img_roc.pkl")
fpr2, tpr2, auc2 = load_and_compute("./clin_roc.pkl")
fpr3, tpr3, auc3 = load_and_compute("./output2/fused_roc.pkl")

# 绘图
plt.figure(figsize=(6, 6))
plt.plot(fpr1, tpr1, lw=3, label=f'Ultrasound (AUC = {auc1:.2f})', color="#c82423" )
plt.plot(fpr2, tpr2, lw=3, label=f'Clinical (AUC = {auc2:.2f})', color="#3480b8")
plt.plot(fpr3, tpr3, lw=3, label=f'Ensemble (AUC = {auc3:.2f})', color="#ffbe7a")
plt.plot([0, 1], [0, 1], linestyle=(0, (6, 6)), color='gray', lw=2)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=12, fontname='Arial')
plt.ylabel('True Positive Rate', fontsize=12, fontname='Arial')
plt.legend(loc='lower right',  bbox_to_anchor=(0.95, 0.1), fontsize=12, prop={'family': 'Arial', 'weight': 'bold'})
plt.grid(False)
plt.tight_layout()
plt.savefig('./output2/three_roc.png', dpi=300)
plt.show()
