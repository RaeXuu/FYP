"""
Confusion matrix figures using training machine test set results.
Run on Mac.

    python plot_confusion_matrix_trainpc.py

Outputs:
    confusion_matrix_diag_trainpc.pdf
    confusion_matrix_sqa_trainpc.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'


def plot_cm(cm, class_names, title, out_path):
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            pct = 100.0 * cm[i, j] / total
            ax.text(j, i,
                    f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    tp = cm[1, 1]; fn = cm[1, 0]; fp = cm[0, 1]; tn = cm[0, 0]
    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / total
    fig.text(0.5, -0.03,
             f"Se={se*100:.1f}%   Sp={sp*100:.1f}%   "
             f"Acc={acc*100:.1f}%   M-Score={(se+sp)/2*100:.1f}%   n={total}",
             ha="center", fontsize=8, color="dimgray")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"{out_path} 已保存")


# ── Diagnostic model (Run 6, training machine test set) ────────────────────
# n=288, Se=94.85%, Sp=83.22%, Acc=85.41%
# n_abnormal=54, n_normal=234  →  TP=51, FN=3, TN=195, FP=39
diag_cm = np.array([
    [195, 39],   # True Normal:   TN=195, FP=39
    [  3, 51],   # True Abnormal: FN=3,   TP=51
])
plot_cm(
    diag_cm,
    class_names=["Normal", "Abnormal"],
    title="Diagnostic Model — Confusion Matrix\n(Run 6, test set, n=288)",
    out_path="confusion_matrix_diag_trainpc.png",
)

# ── SQA model (Run 3, training machine test set) ────────────────────────────
# n=324, Se(Bad)=82.74%, Sp(Good)=80.21%, Acc=80.56%
# n_bad=36, n_good=288  →  TP=30, FN=6, TN=231, FP=57
sqa_cm = np.array([
    [231, 57],  # True Good: TN=231, FP=57
    [  6, 30],  # True Bad:  FN=6,   TP=30
])
plot_cm(
    sqa_cm,
    class_names=["Good Quality", "Bad Quality"],
    title="SQA Model — Confusion Matrix\n(Run 3, test set, n=324)",
    out_path="confusion_matrix_sqa_trainpc.png",
)
