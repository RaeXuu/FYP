"""
Confusion matrix figures for the thesis.
Run on Mac (no Pi or TFLite needed).

    python plot_confusion_matrix.py

Outputs:
    confusion_matrix_diag.pdf
    confusion_matrix_sqa.pdf
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


# ── Diagnostic model (from evaluate.py output) ─────────────────────────────
# evaluated=107, Se=97.9%, Sp=40.0%, Acc=65.4%
# n_abnormal=47, n_normal=60
# TP=46, FN=1, TN=24, FP=36
diag_cm = np.array([
    [24, 36],   # True Normal:   TN=24, FP=36
    [ 1, 46],   # True Abnormal: FN=1,  TP=46
])
plot_cm(
    diag_cm,
    class_names=["Normal", "Abnormal"],
    title="Diagnostic Model — Confusion Matrix\n(INT8, Pi 4B, n=107 evaluated)",
    out_path="confusion_matrix_diag.png",
)

# ── SQA model (from evaluate.py output) ────────────────────────────────────
# evaluated=324, Se(Bad)=84.4%, Sp(Good)=80.1%, Acc=80.6%
# n_bad=35, n_good=289
# TP=30 (Bad correctly identified), FN=5, TN=232, FP=57
sqa_cm = np.array([
    [232, 57],  # True Good: TN=232, FP=57
    [  5, 30],  # True Bad:  FN=5,   TP=30
])
plot_cm(
    sqa_cm,
    class_names=["Good Quality", "Bad Quality"],
    title="SQA Model — Confusion Matrix\n(INT8, Pi 4B, n=324)",
    out_path="confusion_matrix_sqa.png",
)
