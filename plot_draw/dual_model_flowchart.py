"""
Dual-model pipeline flowchart.
  Input (rect) → Model 0 SQA (diamond, decides) → Unacceptable: "unsure" (rect)
                                                  → Acceptable:  Model 1 (rect, inference)
                                                                 → "normal/abnormal" (rect)
Colors inspired by reference image.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(0, 9)
ax.set_ylim(0, 7)
ax.axis('off')

# ── Colour palette ────────────────────────────────────────────────
C_DECISION_FILL = '#D4E6F4'   # light blue  — diamond (Model 0)
C_PROCESS_FILL  = '#D4E6F4'   # light blue  — rectangle (Model 1, Input)
C_REJECT_FILL   = '#FDECEA'   # light red   — "unsure"
C_OUTPUT_FILL   = '#E8F8F0'   # light green — classification output
C_BORDER        = '#2E86C1'   # blue border (all shapes)
C_REJECT_BDR    = '#C0392B'   # red border  — "unsure"
C_OUTPUT_BDR    = '#1E8449'   # green border — output

C_MODEL_TEXT    = '#C0392B'   # red bold — model name text
C_EDGE_LABEL    = '#2471A3'   # blue — arrow labels
C_OUTPUT_TEXT   = '#E67E22'   # orange — output node text
C_INPUT_TEXT    = '#1A252F'   # dark — input text
C_ARROW         = '#E67E22'   # orange arrows


# ── helpers ───────────────────────────────────────────────────────
def rectangle(ax, cx, cy, w, h, label,
              fc=C_PROCESS_FILL, ec=C_BORDER,
              text_color=C_INPUT_TEXT, fs=10.5, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.06',
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2))
    lines = label.split('\n')
    for i, line in enumerate(lines):
        offset = (len(lines) - 1) / 2 * 0.20 - i * 0.20
        ax.text(cx, cy + offset, line, ha='center', va='center',
                fontsize=fs, color=text_color,
                fontweight='bold' if bold else 'normal', zorder=3)


def diamond(ax, cx, cy, w, h, label,
            fc=C_DECISION_FILL, ec=C_BORDER,
            text_color=C_MODEL_TEXT, fs=10.5):
    pts = np.array([[cx, cy+h/2], [cx+w/2, cy],
                    [cx, cy-h/2], [cx-w/2, cy]])
    ax.add_patch(plt.Polygon(pts, closed=True,
                             facecolor=fc, edgecolor=ec,
                             linewidth=1.8, zorder=2))
    lines = label.split('\n')
    for i, line in enumerate(lines):
        offset = (len(lines) - 1) / 2 * 0.24 - i * 0.24
        ax.text(cx, cy + offset, line, ha='center', va='center',
                fontsize=fs, color=text_color,
                fontweight='bold', zorder=3)


def arrow(ax, x0, y0, x1, y1, label='', label_side='right'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=2.0, mutation_scale=14),
                zorder=4)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ha  = 'left'  if label_side == 'right' else 'right'
        dx  =  0.15   if label_side == 'right' else -0.15
        ax.text(mx+dx, my, label, ha=ha, va='center',
                fontsize=9.5, color=C_EDGE_LABEL,
                fontweight='bold', style='italic', zorder=5)


# ── layout ───────────────────────────────────────────────────────
MX = 5.2          # main vertical spine x
RW, RH = 2.3, 0.62    # rectangle w/h
DW, DH = 2.9, 1.15    # diamond w/h

POS = {
    'input':  (MX,   6.15),
    'sqa':    (MX,   4.60),
    'unsure': (1.95, 4.60),
    'diag':   (MX,   2.95),
    'output': (MX,   1.35),
}

# ── nodes ────────────────────────────────────────────────────────
rectangle(ax, *POS['input'],  RW, RH,
          label='Input: .wav', fs=11, bold=True)

diamond(ax, *POS['sqa'], DW, DH,
        label='Model 0:\nQuality Assessment')

rectangle(ax, *POS['unsure'], RW, RH,
          label='"unsure"',
          fc=C_REJECT_FILL, ec=C_REJECT_BDR,
          text_color=C_OUTPUT_TEXT, fs=11, bold=True)

rectangle(ax, *POS['diag'], RW + 0.5, RH + 0.15,
          label='Model 1:\nHeart Sound Classification',
          text_color=C_MODEL_TEXT, fs=10.5, bold=True)

rectangle(ax, *POS['output'], RW + 0.7, RH,
          label='"normal" or "abnormal"',
          fc=C_OUTPUT_FILL, ec=C_OUTPUT_BDR,
          text_color=C_OUTPUT_TEXT, fs=11, bold=True)

# ── arrows ───────────────────────────────────────────────────────
arrow(ax,
      POS['input'][0],  POS['input'][1] - RH/2,
      POS['sqa'][0],    POS['sqa'][1]   + DH/2)

arrow(ax,
      POS['sqa'][0] - DW/2,    POS['sqa'][1],
      POS['unsure'][0] + RW/2, POS['unsure'][1],
      label='Unacceptable', label_side='left')

arrow(ax,
      POS['sqa'][0],  POS['sqa'][1]  - DH/2,
      POS['diag'][0], POS['diag'][1] + (RH+0.15)/2,
      label='Acceptable', label_side='right')

arrow(ax,
      POS['diag'][0],   POS['diag'][1]   - (RH+0.15)/2,
      POS['output'][0], POS['output'][1] + RH/2)

# ── legend ───────────────────────────────────────────────────────
ax.legend(handles=[
    mpatches.Patch(fc=C_DECISION_FILL, ec=C_BORDER,     label='Decision node  (diamond)'),
    mpatches.Patch(fc=C_PROCESS_FILL,  ec=C_BORDER,     label='Process / Data  (rectangle)'),
    mpatches.Patch(fc=C_REJECT_FILL,   ec=C_REJECT_BDR, label='Rejected output'),
    mpatches.Patch(fc=C_OUTPUT_FILL,   ec=C_OUTPUT_BDR, label='Classification output'),
], loc='lower right', fontsize=8.5, framealpha=0.85, edgecolor='#bbbbbb')

ax.set_title('Dual-Model Heart Sound Classification Pipeline',
             fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('fig_dual_model_flowchart.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print("fig_dual_model_flowchart.png 已保存")
