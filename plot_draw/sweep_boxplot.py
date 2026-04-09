import os
import yaml
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

SWEEP_DIR = "/home/agiuser/FypProj/wandb/sweep-cxrbt3c5"

# ---- 读本地 config yaml ----
local_configs = {}
for fname in os.listdir(SWEEP_DIR):
    if not fname.startswith("config-") or not fname.endswith(".yaml"):
        continue
    run_id = fname[len("config-"):-len(".yaml")]
    with open(os.path.join(SWEEP_DIR, fname)) as f:
        raw = yaml.safe_load(f)
    local_configs[run_id] = {k: v["value"] for k, v in raw.items()
                              if isinstance(v, dict) and "value" in v}

print(f"本地 config 数: {len(local_configs)}")

# ---- 拉 W&B summary ----
api = wandb.Api()
sweep = api.sweep("xrjgoole-google/heart-sound-fyp/sweeps/cxrbt3c5")

records = []
for run in sweep.runs:
    run_id  = run.id
    summary = run.summary
    m  = summary.get("best_val_m_score") or summary.get("val/m_score")
    se = summary.get("best_val_se")      or summary.get("val/sensitivity")
    sp = summary.get("best_val_sp")      or summary.get("val/specificity")
    if m is None:
        continue
    cfg = local_configs.get(run_id, {})
    records.append({
        "run_name":    run.name,
        "M-Score":     m,
        "Sensitivity": se,
        "Specificity": sp,
        **cfg,
    })

df = pd.DataFrame(records).dropna(subset=["M-Score"])
print(f"有效 run 数: {len(df)}")
print(df[["n_mels","hop_length","n_fft","overlap","lr","weight_decay"]].describe())

best = df.loc[df["M-Score"].idxmax()]
print(f"\n最优 run: {best['run_name']} | M-Score={best['M-Score']:.4f}")

# ---- 画 2x3 箱形图（按超参数分组） ----
params = [
    ("n_mels",       "Number of Mel bins"),
    ("hop_length",   "Hop length"),
    ("n_fft",        "FFT size"),
    ("overlap",      "Overlap ratio"),
    ("lr",           "Learning rate"),
    ("weight_decay", "Weight decay"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

for ax, (col, xlabel) in zip(axes, params):
    if col not in df.columns or df[col].dropna().empty:
        ax.set_visible(False)
        continue

    groups = sorted(df[col].dropna().unique())
    data   = [df[df[col] == g]["M-Score"].dropna().values for g in groups]

    bp = ax.boxplot(data, labels=[str(g) for g in groups], patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor='#AEC6CF', alpha=0.8),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    # jitter 散点
    for i, g in enumerate(groups, start=1):
        vals = df[df[col] == g]["M-Score"].dropna().values
        ax.scatter(np.random.normal(i, 0.04, len(vals)), vals,
                   alpha=0.4, s=12, color='gray', zorder=3)

    # 最优 run 红星
    if pd.notna(best.get(col)) and best[col] in groups:
        best_idx = groups.index(best[col]) + 1
        ax.plot(best_idx, best["M-Score"], '*', color='crimson',
                markersize=11, zorder=5, label='Best run')
        ax.legend(fontsize=8)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Val M-Score', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(axis='y', alpha=0.35, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(f'Bayesian Sweep: M-Score Distribution by Hyperparameter (n={len(df)} runs)',
             fontsize=10)
plt.tight_layout()
plt.savefig('fig5_sweep_boxplot.png', dpi=300, bbox_inches='tight')
print("\nfig5_sweep_boxplot.png 已保存")