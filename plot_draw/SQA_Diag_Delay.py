import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'

stages = ['Bandpass\nfilter', 'Log-Mel\nspectrogram', 'SQA\nmodel', 'Diagnostic\nmodel']
fp32 = [2.24, 4.73, 13.51, 13.44]
int8 = [2.24, 4.73, 13.46, 13.43]  # bandpass/mel 不走 TFLite，与 FP32 值相同；caption 需说明

x = np.arange(len(stages))
w = 0.28
gap = 0.04
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2 - gap/2, fp32, w, label='FP32', zorder=3)
ax.bar(x + w/2 + gap/2, int8, w, label='INT8', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=8)
ax.set_ylabel('Latency (ms)', fontsize=9)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.35, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig6_2_latency.png', dpi=300, bbox_inches='tight')
print("fig6_2_latency.png 已保存")
