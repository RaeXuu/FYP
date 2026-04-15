import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'

stages = ['Bandpass filter', 'Log-Mel spectrogram', 'SQA model', 'Diagnostic model']
fp32 = [2.24, 4.73, 13.51, 13.44]
int8 = [2.24, 4.73, 13.46, 13.43]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

y = np.arange(2)
h = 0.3
fig, ax = plt.subplots(figsize=(8, 3.8))

left_fp32 = 0
left_int8 = 0
for i, stage in enumerate(stages):
    ax.barh(1, fp32[i], h, left=left_fp32, color=colors[i],
            label=stage, zorder=3)
    ax.barh(0, int8[i], h, left=left_int8, color=colors[i],
            zorder=3)
    if fp32[i] > 1:
        ax.text(left_fp32 + fp32[i]/2, 1, f'{fp32[i]:.2f}',
                ha='center', va='center', fontsize=7.5, color='white')
    if int8[i] > 1:
        ax.text(left_int8 + int8[i]/2, 0, f'{int8[i]:.2f}',
                ha='center', va='center', fontsize=7.5, color='white')
    left_fp32 += fp32[i]
    left_int8 += int8[i]

ax.text(left_fp32 + 0.3, 1, f'{sum(fp32):.2f} ms', va='center', fontsize=7.5, color='#333333')
ax.text(left_int8 + 0.3, 0, f'{sum(int8):.2f} ms', va='center', fontsize=7.5, color='#333333')

ax.set_yticks(y)
ax.set_yticklabels(['INT8', 'FP32'], fontsize=8)
ax.set_xlabel('Latency (ms)', fontsize=9)
ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15))
ax.grid(axis='x', alpha=0.35, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig6_2_latency.png', dpi=300, bbox_inches='tight')
print("fig6_2_latency.png 已保存")
