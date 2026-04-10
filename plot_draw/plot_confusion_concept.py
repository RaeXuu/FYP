"""
Conceptual confusion matrix figure for heart sound classification.
Left: matrix with row/column metrics (reference style)
Right: equivalences and key formulas used in this paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis('off')

GREEN = '#2e7d32'
RED   = '#c62828'
WHITE = 'white'

# ── cells: (left, bottom, width, height) ────────────────────────
cells = {
    'TP': (0.8, 2.9, 1.7, 1.8),
    'FN': (2.6, 2.9, 1.7, 1.8),
    'FP': (0.8, 1.0, 1.7, 1.8),
    'TN': (2.6, 1.0, 1.7, 1.8),
}
colors = {'TP': GREEN, 'FN': RED, 'FP': RED, 'TN': GREEN}

for key, (x, y, w, h) in cells.items():
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0.0",
        facecolor=colors[key], edgecolor='black', linewidth=2,
        alpha=0.88, zorder=2))

# ── cell text ────────────────────────────────────────────────────
cell_texts = {
    'TP': ("True Positive (TP)",   "Abnormal correctly\nidentified as Abnormal"),
    'FN': ("False Negative (FN)",  "Abnormal misclassified\nas Normal"),
    'FP': ("False Positive (FP)",  "Normal misclassified\nas Abnormal"),
    'TN': ("True Negative (TN)",   "Normal correctly\nidentified as Normal"),
}
centers = {'TP': (1.65, 3.80), 'FN': (3.45, 3.80),
           'FP': (1.65, 1.90), 'TN': (3.45, 1.90)}

for key, (title, desc) in cell_texts.items():
    cx, cy = centers[key]
    ax.text(cx, cy + 0.28, title,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=WHITE, zorder=3)
    ax.text(cx, cy - 0.18, desc,
            ha='center', va='center', fontsize=8,
            color=WHITE, style='italic', zorder=3, linespacing=1.4)

# ── "Predicted" arrow top ────────────────────────────────────────
ax.annotate('', xy=(4.35, 5.10), xytext=(0.75, 5.10),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(2.55, 5.28, 'Predicted', ha='center', fontsize=10, fontweight='bold')
ax.text(1.65, 4.95, 'Abnormal', ha='center', fontsize=9)
ax.text(3.45, 4.95, 'Normal',   ha='center', fontsize=9)

# ── "Actual" arrow left ──────────────────────────────────────────
ax.annotate('', xy=(0.65, 0.95), xytext=(0.65, 4.90),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(0.32, 2.85, 'Actual', ha='center', fontsize=10,
        fontweight='bold', rotation=90)
ax.text(0.72, 3.80, 'Abnormal', ha='center', fontsize=9, rotation=90)
ax.text(0.72, 1.90, 'Normal',   ha='center', fontsize=9, rotation=90)

# ── RIGHT of matrix: row-wise metrics ───────────────────────────
ax.text(4.38, 3.80,
        r'$Recall = \frac{TP}{TP+FN}$',
        ha='left', va='center', fontsize=10.5)
ax.text(4.38, 1.90,
        r'$Specificity = \frac{TN}{TN+FP}$',
        ha='left', va='center', fontsize=10.5)

# ── BELOW matrix: column-wise metrics ───────────────────────────
ax.text(1.65, 0.55,
        r'$Precision = \frac{TP}{TP+FP}$',
        ha='center', va='center', fontsize=9.5)
ax.text(3.45, 0.55,
        r'$NPV = \frac{TN}{TN+FN}$',
        ha='center', va='center', fontsize=9.5)

# ── vertical divider ─────────────────────────────────────────────
ax.plot([6.0, 6.0], [0.0, 5.50], color='#bbbbbb', lw=1.2, ls='--')

# ── RIGHT panel: equivalences + paper metrics ────────────────────
ax.text(6.6, 5.00,
        r'$Recall = Sensitivity = Se = \frac{TP}{TP+FN}$',
        ha='left', va='center', fontsize=10.5)
ax.text(6.6, 3.75,
        r'$Specificity = Sp = \frac{TN}{TN+FP}$',
        ha='left', va='center', fontsize=10.5)
ax.text(6.6, 2.50,
        r'$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$',
        ha='left', va='center', fontsize=10.5)
ax.text(6.6, 1.25,
        r'$M\text{-}Score = \frac{Se+Sp}{2}$',
        ha='left', va='center', fontsize=10.5)

plt.tight_layout()
plt.savefig('fig_confusion_concept.png', dpi=300, bbox_inches='tight', facecolor='white')
print("fig_confusion_concept.png 已保存")
