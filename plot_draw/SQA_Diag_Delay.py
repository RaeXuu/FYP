import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

stages = ["Bandpass filter", "Log-Mel spectrogram", "SQA model", "Diagnostic model"]
fp32     = [2.24, 4.73, 14.2, 14.1]
int8_dyn = [2.24, 4.73, 13.8, 13.8]
int8_full= [2.24, 4.73,  8.7,  8.7]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

y = np.arange(3)
h = 0.28
fig, ax = plt.subplots(figsize=(9, 4.2))

rows = [
    (2, fp32,      "FP32"),
    (1, int8_dyn,  "INT8 Dynamic"),
    (0, int8_full, "INT8 Full Integer"),
]

for row_idx, (yi, data, label) in enumerate(rows):
    left = 0
    for i, stage in enumerate(stages):
        bar = ax.barh(yi, data[i], h, left=left, color=colors[i],
                      label=stage if row_idx == 0 else "", zorder=3)
        if data[i] > 1.0:
            ax.text(left + data[i] / 2, yi, f"{data[i]:.2f}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        left += data[i]
    total = sum(data)
    ax.text(left + 0.3, yi, f"{total:.1f} ms",
            va="center", fontsize=8, color="#333333")

ax.set_yticks(y)
ax.set_yticklabels(["INT8 Full Integer", "INT8 Dynamic", "FP32"], fontsize=9)
ax.set_xlabel("Latency (ms)", fontsize=9)
ax.set_xlim(0, 40)

handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles, labels_leg, fontsize=8, ncol=4,
          loc="upper center", bbox_to_anchor=(0.5, 1.15))
ax.grid(axis="x", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig6_2_latency.png", dpi=300, bbox_inches="tight")
print("fig6_2_latency.png saved")
