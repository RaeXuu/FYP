"""
Confusion matrix figures using training machine test set evaluation results.
Run on Mac.

    python plot_confusion_matrix_trainpc.py

Outputs:
    confusion_matrix_diag_trainpc.png
    confusion_matrix_sqa_trainpc.png
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


# ── Diagnostic model (evaluate_tflite.py, FP32, per-slice) ─────────────────
# n=6273, Se=96.45%, Sp=80.25%, Acc=83.31%, M-Score=88.35%
# n_abnormal=1184 (TP+FN), n_normal=5089 (TN+FP)
# TN=4084, FP=1005, FN=42, TP=1142
diag_cm = np.array([
    [4084, 1005],   # True Normal:   TN=4084, FP=1005
    [  42, 1142],   # True Abnormal: FN=42,   TP=1142
])
plot_cm(
    diag_cm,
    class_names=["Normal", "Abnormal"],
    title="Diagnostic Model — Confusion Matrix\n(FP32, training machine, n=6273 slices)",
    out_path="confusion_matrix_diag_trainpc.png",
)

# ── SQA model (evaluate_tflite.py, FP32, per-slice) ────────────────────────
# n=6726, Se(Bad)=90.44%, Sp(Good)=71.91%, Acc=73.24%, M-Score=81.18%
# n_bad=481 (TP+FN), n_good=6245 (TN+FP)
# TN=4491, FP=1754, FN=46, TP=435
sqa_cm = np.array([
    [4491, 1754],  # True Good: TN=4491, FP=1754
    [  46,  435],  # True Bad:  FN=46,   TP=435
])
plot_cm(
    sqa_cm,
    class_names=["Good Quality", "Bad Quality"],
    title="SQA Model — Confusion Matrix\n(FP32, training machine, n=6726 slices)",
    out_path="confusion_matrix_sqa_trainpc.png",
)
