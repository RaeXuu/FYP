import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
se = [0.9890, 0.9840, 0.9764, 0.9688, 0.9569, 0.9417, 0.9231, 0.8547, 0.7652]
sp = [0.7787, 0.7860, 0.7947, 0.8031, 0.8135, 0.8218, 0.8332, 0.8562, 0.8915]
ms = [0.8839, 0.8850, 0.8855, 0.8859, 0.8852, 0.8817, 0.8782, 0.8554, 0.8284]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, se, 'o-', label='Sensitivity')
ax.plot(thresholds, sp, 's-', label='Specificity')
ax.plot(thresholds, ms, '^-', label='M-Score')
ax.axvline(0.50, color='gray', linestyle='--', alpha=0.6, label='Default (0.50)')
ax.set_xlabel('Threshold', fontsize=9)
ax.set_ylabel('Score', fontsize=9)
ax.tick_params(axis='both', labelsize=8)
ax.legend(fontsize=8)
ax.grid(alpha=0.35, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig5_3_threshold.png', dpi=300, bbox_inches='tight')
print("fig5_3_threshold.png 已保存")
