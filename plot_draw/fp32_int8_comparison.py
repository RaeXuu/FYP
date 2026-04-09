import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'

labels = ['Diagnostic\nModel', 'SQA\nModel']
fp32_sizes = [302.8, 302.8]
int8_sizes = [144.7, 144.7]

x = np.arange(len(labels))
w = 0.28
gap = 0.04
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - w/2 - gap/2, fp32_sizes, w, label='FP32', zorder=3)
ax.bar(x + w/2 + gap/2, int8_sizes, w, label='INT8', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Model Size (KB)', fontsize=9)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)

for i in range(len(labels)):
    ax.annotate('−52.2%', xy=(x[i], int8_sizes[i] + 5),
                ha='center', fontsize=7.5, color='gray')

plt.tight_layout()
plt.savefig('fig6_3_model_size.png', dpi=300, bbox_inches='tight')
print("fig6_3_model_size.png 已保存")
