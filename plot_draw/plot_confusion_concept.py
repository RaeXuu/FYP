"""
Conceptual confusion matrix figure for heart sound classification.
Left: matrix with row/column metrics (reference style)
Right: equivalences and key formulas used in this paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(10, 5))   # 2:1 匹配 xlim/ylim 比例
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.set_aspect('equal')                     # 每数据单位物理尺寸相同，消除畸变
ax.axis('off')

GREEN = '#2e7d32'
RED   = '#c62828'
WHITE = 'white'

# ── cells: (left, bottom, width, height) ── width=height=1.6 保证正方形 ──
# 水平间隙 = 2.6-(0.8+1.6) = 0.2，垂直间隙 = 2.6-(0.8+1.6) = 0.2，四周一致
cells = {
    'TP': (0.8, 2.6, 1.6, 1.6),
    'FN': (2.6, 2.6, 1.6, 1.6),
    'FP': (0.8, 0.8, 1.6, 1.6),
    'TN': (2.6, 0.8, 1.6, 1.6),
}
colors = {'TP': GREEN, 'FN': RED, 'FP': RED, 'TN': GREEN}

for key, (x, y, w, h) in cells.items():
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0.0",
        facecolor=colors[key], edgecolor='black', linewidth=2,
        alpha=0.88, zorder=2))

# ── cell text ─────────────────────────────────────────────────────
cell_texts = {
    'TP': ("True Positive (TP)",   "Abnormal correctly\nidentified as Abnormal"),
    'FN': ("False Negative (FN)",  "Abnormal misclassified\nas Normal"),
    'FP': ("False Positive (FP)",  "Normal misclassified\nas Abnormal"),
    'TN': ("True Negative (TN)",   "Normal correctly\nidentified as Normal"),
}
# 色块中心 = left+w/2, bottom+h/2
centers = {'TP': (1.60, 3.40), 'FN': (3.40, 3.40),
           'FP': (1.60, 1.60), 'TN': (3.40, 1.60)}

for key, (title, desc) in cell_texts.items():
    cx, cy = centers[key]
    ax.text(cx, cy + 0.28, title,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=WHITE, zorder=3)
    ax.text(cx, cy - 0.18, desc,
            ha='center', va='center', fontsize=8,
            color=WHITE, style='italic', zorder=3, linespacing=1.4)

# ── "Predicted" arrow top ─────────────────────────────────────────
# 色块顶部 = 2.6+1.6 = 4.2；箭头在 4.45，列标签在 4.30
ax.annotate('', xy=(4.25, 4.45), xytext=(0.75, 4.45),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(2.50, 4.57, 'Predicted', ha='center', fontsize=10, fontweight='bold')
ax.text(1.60, 4.30, 'Abnormal', ha='center', fontsize=9)
ax.text(3.40, 4.30, 'Normal',   ha='center', fontsize=9)

# ── "Actual" arrow left ───────────────────────────────────────────
# 色块从 y=0.8 到 y=4.2；箭头贴齐两端
ax.annotate('', xy=(0.45, 0.8), xytext=(0.45, 4.20),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(0.28, 2.50, 'Actual', ha='center', fontsize=10,
        fontweight='bold', rotation=90)
ax.text(0.58, 3.40, 'Abnormal', ha='center', fontsize=9, rotation=90)
ax.text(0.58, 1.60, 'Normal',   ha='center', fontsize=9, rotation=90)

# ── RIGHT of matrix: row-wise metrics ────────────────────────────
ax.text(4.30, 3.40,
        r'$Recall = \frac{TP}{TP+FN}$',
        ha='left', va='center', fontsize=10.5)
ax.text(4.30, 1.60,
        r'$Specificity = \frac{TN}{TN+FP}$',
        ha='left', va='center', fontsize=10.5)

# ── BELOW matrix: column-wise metrics ────────────────────────────
ax.text(1.60, 0.52,
        r'$Precision = \frac{TP}{TP+FP}$',
        ha='center', va='center', fontsize=10.5)
ax.text(3.40, 0.52,
        r'$NPV = \frac{TN}{TN+FN}$',
        ha='center', va='center', fontsize=10.5)

# ── vertical divider ──────────────────────────────────────────────
ax.plot([6.0, 6.0], [0.0, 5.0], color='#bbbbbb', lw=1.2, ls='--')

# ── RIGHT panel: equivalences + paper metrics ─────────────────────
ax.text(6.2, 4.50,
        r'$Recall = Sensitivity = Se = \frac{TP}{TP+FN}$',
        ha='left', va='center', fontsize=12)
ax.text(6.2, 3.25,
        r'$Specificity = Sp = \frac{TN}{TN+FP}$',
        ha='left', va='center', fontsize=12)
ax.text(6.2, 2.00,
        r'$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$',
        ha='left', va='center', fontsize=12)
ax.text(6.2, 0.75,
        r'$M\text{-}Score = \frac{Se+Sp}{2}$',
        ha='left', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('fig_confusion_concept.png', dpi=300, bbox_inches='tight', facecolor='white')
print("fig_confusion_concept.png 已保存")
