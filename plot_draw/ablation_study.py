import matplotlib.pyplot as plt
import numpy as np

configs  = ['A: Baseline', 'B: +Wider', 'C: +CoordAtt', 'D: +Residual']
m_scores = [0.8851, 0.8896, 0.8869, 0.8912]
se       = [0.9654, 0.9595, 0.9383, 0.9797]
sp       = [0.8049, 0.8198, 0.8355, 0.8027]

x   = np.arange(len(configs))
w   = 0.22   # 柱子变窄
gap = 0.03   # 柱子之间加间距

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(10, 5))

bars_m  = ax.bar(x - w - gap, m_scores, w, label='M-Score',     zorder=3)
bars_se = ax.bar(x,            se,       w, label='Sensitivity', zorder=3)
bars_sp = ax.bar(x + w + gap,  sp,       w, label='Specificity', zorder=3)

# 数值标注
for bars in [bars_m, bars_se, bars_sp]:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=7.5, color='#333333'
        )

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylim(0.75, 1.02)
ax.set_ylabel('Score', fontsize=9)
ax.set_title('Ablation Study: Model Configuration Comparison', fontsize=10, pad=12)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(axis='y', alpha=0.35, linestyle='--', zorder=0)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig5_2_ablation.png', dpi=300, bbox_inches='tight')
print("fig5_2_ablation.png 已保存")
