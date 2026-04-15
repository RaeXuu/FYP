import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
run = api.run("xrjgoole-google/heart-sound-fyp/ogymmyh3")

df = run.history(keys=["epoch", "train/loss", "val/m_score"], pandas=True)
df = df.dropna(subset=["train/loss", "val/m_score"]).sort_values("_step")

test_mscore = run.summary.get("test_m_score", None)

epochs     = df["_step"]
train_loss = df["train/loss"]
val_mscore = df["val/m_score"]

plt.rcParams['font.family'] = 'serif'
fig, ax1 = plt.subplots(figsize=(8, 4))

color_loss = 'steelblue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color_loss)
ax1.plot(epochs, train_loss, color=color_loss, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)

ax2 = ax1.twinx()
color_ms = 'darkorange'
ax2.set_ylabel('Val M-Score', color=color_ms)
ax2.plot(epochs, val_mscore, color=color_ms, linestyle='--', label='Val M-Score')
ax2.tick_params(axis='y', labelcolor=color_ms)

if test_mscore is not None:
    ax2.axhline(test_mscore, color='crimson', linestyle=':', linewidth=1.5,
                label=f'Test M-Score ({test_mscore:.4f})')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_7_Run3_training_curve.png', dpi=300, bbox_inches='tight')
print("fig5_7_Run3_training_curve.pdf 已保存")
