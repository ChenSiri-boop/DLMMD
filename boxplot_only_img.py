import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

# 1. 数据准备与替换
img_df = pd.read_csv('./output3/3_img_roc_test.csv')
probs_a = img_df['prob'].values
labels = img_df['label'].values

# 创建适合箱型图的DataFrame
df = pd.DataFrame({
    'Predicted Probability': probs_a,
    'Group': ['Metastasis' if label == 1 else 'No Metastasis' for label in labels]
})

# 2. 创建图形
plt.figure(figsize=(6, 6))

# 3. 绘制箱型图
# ax = sns.boxplot(x='Group', y='Predicted Probability', data=df,
#                 palette=['#1f77b4', '#ff7f0e'],  # 蓝色和橙色
#                 width=0.4, linewidth=1.5)
ax = sns.boxplot(x='Group', y='Predicted Probability', data=df,
                palette=['#1f77b4', '#ff7f0e'],
                width=0.5,  # 增加箱型宽度来减小间距
                linewidth=1.5,
                gap=0.1)    # 控制组间间距(某些seaborn版本支持)
# 4. 添加散点
sns.stripplot(x='Group', y='Predicted Probability', data=df,
             palette=['#1f77b4', '#ff7f0e'],
             size=6, jitter=0.1, linewidth=0.5,
             edgecolor='gray', alpha=0.7)

# 5. 计算统计显著性
group1 = df[df['Group'] == 'No Metastasis']['Predicted Probability']
group2 = df[df['Group'] == 'Metastasis']['Predicted Probability']
stat, p_value = mannwhitneyu(group1, group2)

# 6. 添加统计显著性标记
y_max = df['Predicted Probability'].max() + 0.05
plt.plot([0, 0, 1, 1], [y_max-0.02, y_max, y_max, y_max-0.02],
        lw=1.5, color='black')
plt.text(0.5, y_max+0.02, f'P = {p_value:.7f}',  # 保留3位小数
        ha='center', va='bottom', fontsize=12)

# 7. 设置标题和标签
# plt.title('Predicted Probability Distribution by Metastasis Status', fontsize=14, pad=20)
plt.xlabel('')
plt.ylabel('Predicted Probability', fontsize=12)

# 8. 调整y轴范围
plt.ylim(min(df['Predicted Probability']) - 0.05,
        max(df['Predicted Probability']) + 0.15)

# 9. 调整布局
plt.tight_layout()

# 10. 保存和显示
plt.savefig('./output3/boxplot.png', dpi=300, bbox_inches='tight')
plt.show()