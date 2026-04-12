import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

# 更新后的数据（基于日志提取）
# 更新后的数据（基于最新运行结果，最优阈值为 0.50）
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
se = [0.9882, 0.9856, 0.9772, 0.9713, 0.9645, 0.9451, 0.9215, 0.8910, 0.8539, 0.7990, 0.7095]
sp = [0.7703, 0.7797, 0.7868, 0.7947, 0.8025, 0.8127, 0.8259, 0.8389, 0.8554, 0.8768, 0.9037]
ms = [0.8792, 0.8827, 0.8820, 0.8830, 0.8835, 0.8789, 0.8737, 0.8650, 0.8546, 0.8379, 0.8066]
# 更新后的数据（基于日志提取）
# thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
# se = [0.9480, 0.9314, 0.9210, 0.9168, 0.9044, 0.8898, 0.8773, 0.8732, 0.8565, 0.8420, 0.8212]
# sp = [0.6665, 0.6813, 0.6934, 0.7078, 0.7191, 0.7311, 0.7419, 0.7553, 0.7667, 0.7808, 0.7990]
# ms = [0.8072, 0.8064, 0.8072, 0.8123, 0.8118, 0.8105, 0.8096, 0.8143, 0.8116, 0.8114, 0.8101]


fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, se, 'o-', label='Sensitivity')
ax.plot(thresholds, sp, 's-', label='Specificity')
ax.plot(thresholds, ms, '^-', label='M-Score')
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.6, label='Best (0.5)')
ax.set_xlabel('Threshold', fontsize=9)
ax.set_ylabel('Score', fontsize=9)
ax.tick_params(axis='both', labelsize=8)
ax.legend(fontsize=8)
ax.grid(alpha=0.35, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig5_3_threshold_diag.png', dpi=300, bbox_inches='tight')
print("fig5_3_threshold_diag.png 已保存")
# plt.savefig('fig5_3_threshold_sqa.png', dpi=300, bbox_inches='tight')
# print("fig5_3_threshold_sqa.png 已保存")
