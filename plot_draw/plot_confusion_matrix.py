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


# ── Diagnostic model (Pi 4B evaluate.py --mode diag, INT8, per-slice) ───────
# evaluated=6273 slices, Se=91.7%, Sp=82.3%, Acc=84.1%, M-Score=87.0%
# n_normal=5089 slices, n_abnormal=1184 slices
# Se=91.7%  → TP=round(0.917×1184)=1086, FN=98
# Sp=82.3%  → TN=round(0.823×5089)=4188, FP=901
diag_cm = np.array([
    [4188,  901],   # True Normal:   TN=4188, FP=901
    [  98, 1086],   # True Abnormal: FN=98,   TP=1086
])
plot_cm(
    diag_cm,
    class_names=["Normal", "Abnormal"],
    title="Diagnostic Model — Confusion Matrix\n(INT8, Pi 4B, n=6273 slices)",
    out_path="confusion_matrix_diag.png",
)

# ── SQA model (Pi 4B evaluate.py --mode sqa, INT8) ─────────────────────────
# evaluated=6726 slices, Se(Bad)=90.2%, Sp(Good)=73.7%, Acc=74.9%, M-Score=82.0%
# TP=434, TN=4605, FP=1640, FN=47
sqa_cm = np.array([
    [4605, 1640],  # True Good: TN=4605, FP=1640
    [  47,  434],  # True Bad:  FN=47,   TP=434
])
plot_cm(
    sqa_cm,
    class_names=["Good Quality", "Bad Quality"],
    title="SQA Model — Confusion Matrix\n(INT8, Pi 4B, n=6726 slices)",
    out_path="confusion_matrix_sqa.png",
)

# ── Coupled system: SQA-gated Diagnostic (Pi 4B --mode both, FP32) ──────────
# evaluated=143/288 recordings (skipped=145 by SQA gate)
# Se=100.0%, Sp=50.5%, Acc=67.8%, M-Score=75.3%
# n_ab=50, n_nor=93
# Se=100.0% → TP=50, FN=0
# Sp=50.5%  → TN=round(0.505×93)=47, FP=46
coupled_fp32_cm = np.array([
    [47, 46],   # True Normal:   TN=47, FP=46
    [ 0, 50],   # True Abnormal: FN=0,  TP=50
])
plot_cm(
    coupled_fp32_cm,
    class_names=["Normal", "Abnormal"],
    title="Coupled System — Confusion Matrix\n(FP32, Pi 4B, n=143 recordings, skipped=145)",
    out_path="confusion_matrix_coupled_fp32.png",
)

# ── Coupled system: SQA-gated Diagnostic (Pi 4B --mode both, INT8) ──────────
# evaluated=142/288 recordings (skipped=146 by SQA gate)
# Se=100.0%, Sp=50.0%, Acc=67.6%, M-Score=75.0%
# n_ab=50, n_nor=92
# Se=100.0% → TP=50, FN=0
# Sp=50.0%  → TN=round(0.5×92)=46, FP=46
coupled_int8_cm = np.array([
    [46, 46],   # True Normal:   TN=46, FP=46
    [ 0, 50],   # True Abnormal: FN=0,  TP=50
])
plot_cm(
    coupled_int8_cm,
    class_names=["Normal", "Abnormal"],
    title="Coupled System — Confusion Matrix\n(INT8, Pi 4B, n=142 recordings, skipped=146)",
    out_path="confusion_matrix_coupled_int8.png",
)
