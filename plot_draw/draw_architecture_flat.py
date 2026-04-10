"""
Flat-style architecture diagram for LightweightCNN + CoordAtt.
Style mirrors the block-diagram format commonly used in ML papers.
Output: fig4_1_architecture_flat.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(19, 7))
ax.set_xlim(0, 19.5)
ax.set_ylim(0, 7)
ax.axis('off')

# ── Colour palette (muted, harmonious) ───────────────────────────────────
C_INPUT_BG   = "#758188"   # blue-grey slate
C_CONV       = "#9499C7"   # deep blue (border / inner block)
C_BN         = '#90CAF9'   # light blue
C_NOPOOLING  = '#FFCDD2'   # soft pink/red
C_DW         = "#549FEA"   # medium blue
C_COORDATT   = "#EFAC68"   # warm amber (accent)
C_PW         = '#64B5F6'   # sky blue
C_BNRELU     = '#BBDEFB'   # very light blue
C_MAXPOOL    = "#E88E8D"   # red
C_GAP_BG     = '#F3E5F5'   # very light violet
C_GAP_BDR    = '#7B1FA2'   # purple
C_CLS_BG     = '#E8F5E9'   # very light green
C_CLS_BDR    = '#2E7D32'   # dark green
C_DROPOUT    = '#FFF9C4'   # pale yellow
C_FC         = '#C8E6C9'   # light green

CY = 3.5  # vertical centre of everything

# ── Helpers ───────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, fc, ec='white', lw=1.5, pad=0.04):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad={pad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)

def txt(ax, cx, cy, s, fs=11, color='white', bold=False):
    ax.text(cx, cy, s, ha='center', va='center', fontsize=fs,
            color=color, fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=4)

def arrow(ax, x1, x2, y=CY):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.8),
                zorder=5)

def section_header(ax, cx, top_y, s, color):
    ax.text(cx, top_y, s, ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=color, multialignment='center', zorder=4)

# ── 0  Input block ────────────────────────────────────────────────────────
IX, IW, IH = 1.0, 1.05, 3.1
rbox(ax, IX, CY, IW, IH, C_INPUT_BG, ec='#888888', lw=1.5)
txt(ax, IX, CY, 'Input\nLog-Mel\nSpectrogram', fs=11)
ax.text(IX, CY - IH/2 - 0.15, 'Shape: (1 × 64 × 64)',
        ha='center', va='top', fontsize=12, color='#444444')

# ── 1  Standard Convolution block ────────────────────────────────────────
SX, SW, SH = 3.1, 1.6, 3.1
rbox(ax, SX, CY, SW, SH, '#d6eaf8', ec=C_CONV, lw=2, pad=0.03)
section_header(ax, SX, CY + SH/2 + 0.08, 'Standard\nConvolution', C_CONV)

rw = 1.38
for ry, fc, label, tc in [
    (CY + 0.72, C_CONV,      'Conv2D (3×3), 32', '#1a2b3c'),
    (CY + 0.05, C_BN,        'BN + ReLU',        '#1a2b3c'),
    (CY - 0.62, C_NOPOOLING, 'No Pooling',       '#1a2b3c'),
]:
    rbox(ax, SX, ry, rw, 0.54, fc, pad=0.04)
    txt(ax, SX, ry, label, color=tc)

ax.text(SX, CY - SH/2 - 0.15, '32 Channels',
        ha='center', va='top', fontsize=12, color='#555555')

arrow(ax, IX + IW/2 + 0.05, SX - SW/2 - 0.05)

# ── 2–4  DSC Blocks (with CoordAtt) ──────────────────────────────────────
DSC_ROWS = [
    (C_DW,       'Depthwise\nConv (3×3)', '#1a2b3c'),
    (C_BNRELU,   'BN + ReLU',             '#1a2b3c'),
    (C_PW,       'Pointwise\nConv (1×1)', '#1a2b3c'),
    (C_BNRELU,   'BN + ReLU',             '#1a2b3c'),
    (C_COORDATT, 'CoordAtt',              '#1a2b3c'),
    (C_MAXPOOL,  'Max Pooling',           '#1a2b3c'),
]

BW, BH   = 1.7, 4.3
ROW_H    = 0.48
ROW_GAP  = 0.06
N        = len(DSC_ROWS)
ROWS_TOT = N * ROW_H + (N - 1) * ROW_GAP

DSC_XS    = [5.4, 7.7, 10.0]
DSC_CHANS = [64, 128, 256]
DSC_LABELS = [
    'Depthwise Separable\nConv Block 1',
    'Depthwise Separable\nConv Block 2',
    'Depthwise Separable\nConv Block 3',
]

for bx, ch, title in zip(DSC_XS, DSC_CHANS, DSC_LABELS):
    rbox(ax, bx, CY, BW, BH, '#eaf4fb', ec='#2471a3', lw=2, pad=0.03)
    section_header(ax, bx, CY + BH/2 + 0.08, title, '#1a5276')

    top = CY + ROWS_TOT/2 - ROW_H/2
    for i, (fc, lbl, tc) in enumerate(DSC_ROWS):
        ry = top - i * (ROW_H + ROW_GAP)
        rbox(ax, bx, ry, BW - 0.22, ROW_H, fc, pad=0.04)
        txt(ax, bx, ry, lbl, color=tc)

    ax.text(bx, CY - BH/2 - 0.15, f'{ch} Channels',
            ha='center', va='top', fontsize=12, color='#555555')

arrow(ax, SX + SW/2 + 0.05,       DSC_XS[0] - BW/2 - 0.05)
arrow(ax, DSC_XS[0] + BW/2 + 0.05, DSC_XS[1] - BW/2 - 0.05)
arrow(ax, DSC_XS[1] + BW/2 + 0.05, DSC_XS[2] - BW/2 - 0.05)

# ── 5  Global Average Pooling ─────────────────────────────────────────────
GX, GW, GH = 12.8, 1.55, 2.5
rbox(ax, GX, CY, GW, GH, C_GAP_BG, ec=C_GAP_BDR, lw=2, pad=0.03)
section_header(ax, GX, CY + GH/2 + 0.08, 'Global\nAverage Pool', C_GAP_BDR)
txt(ax, GX, CY + 0.38, 'Adaptive\nAvgPool2D', fs=12, color='#4a235a')
txt(ax, GX, CY - 0.38, '8×8  →  1×1', fs=12, color='#6a1e8a')

arrow(ax, DSC_XS[2] + BW/2 + 0.05, GX - GW/2 - 0.05)

# ── 6  Classification Head ────────────────────────────────────────────────
CX2, CW2, CH2 = 15.2, 1.65, 2.9
rbox(ax, CX2, CY, CW2, CH2, C_CLS_BG, ec=C_CLS_BDR, lw=2, pad=0.03)
section_header(ax, CX2, CY + CH2/2 + 0.08, 'Classification\nHead', C_CLS_BDR)

rbox(ax, CX2, CY + 0.55, CW2 - 0.2, 0.54, C_DROPOUT, pad=0.04)
txt(ax, CX2, CY + 0.55, 'Dropout (0.3)', color='#5D4000', fs=11)

rbox(ax, CX2, CY - 0.18, CW2 - 0.2, 0.54, C_FC, pad=0.04)
txt(ax, CX2, CY - 0.18, 'Fully Connected', color='#1a5e20', fs=11)

ax.text(CX2, CY - 0.72, 'Input: 256',  ha='center', va='center', fontsize=12, color='#555')
ax.text(CX2, CY - 1.02, 'Output: 2',   ha='center', va='center', fontsize=12, color='#555')

arrow(ax, GX + GW/2 + 0.05, CX2 - CW2/2 - 0.05)

# ── 7  Output label ───────────────────────────────────────────────────────
OX = CX2 + CW2/2 + 0.1
arrow(ax, OX, OX + 0.85)
ax.text(OX + 0.95, CY + 0.18, 'Normal',   ha='left', va='center',
        fontsize=12, color='#2E7D32', fontweight='bold')
ax.text(OX + 0.95, CY - 0.18, 'Abnormal', ha='left', va='center',
        fontsize=12, color='#C62828', fontweight='bold')

# ── Legend ────────────────────────────────────────────────────────────────
ax.legend(handles=[
    mpatches.Patch(fc=C_COORDATT,  ec='white', label='CoordAtt (positional attention)'),
    mpatches.Patch(fc=C_DW,        ec='white', label='Depthwise Conv'),
    mpatches.Patch(fc=C_PW,        ec='white', label='Pointwise Conv'),
    mpatches.Patch(fc=C_MAXPOOL,   ec='white', label='Max Pooling'),
], loc='lower left', bbox_to_anchor=(0.01, 0.01),
   fontsize=10, framealpha=0.85, edgecolor='#cccccc')

ax.set_title('LightweightCNN — Detailed Architecture',
             fontsize=14, fontweight='bold', pad=14)

plt.tight_layout()
plt.savefig('fig4_1_architecture_flat.png', dpi=300, bbox_inches='tight',
            facecolor='white')
print("Saved: fig4_1_architecture_flat.png")