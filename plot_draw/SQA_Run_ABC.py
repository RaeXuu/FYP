import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'

runs     = ['Run 1\n(baseline)', 'Run 2\n(+class weight)', 'Run 3\n(+dropout 0.5)']
m_scores = [0.8046, 0.8102, 0.8152]
se       = [0.7173, 0.7651, 0.8274]
sp       = [0.8919, 0.8554, 0.8029]

x   = np.arange(len(runs))
w   = 0.22
gap = 0.03

fig, ax = plt.subplots(figsize=(8, 5))
bars_m  = ax.bar(x - w - gap, m_scores, w, label='M-Score',             zorder=3)
bars_se = ax.bar(x,            se,       w, label='Sensitivity (Bad)',   zorder=3)
bars_sp = ax.bar(x + w + gap,  sp,       w, label='Specificity (Good)',  zorder=3)

for bars in [bars_m, bars_se, bars_sp]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(runs, fontsize=8)
ax.set_ylim(0.70, 0.95)
ax.set_ylabel('Score', fontsize=9)
ax.set_title('SQA Model Training Progress', fontsize=10, pad=10)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.35, linestyle='--', zorder=0)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig5_4_sqa_runs.png', dpi=300, bbox_inches='tight')
print("fig5_4_sqa_runs.png 已保存")
