"""
System architecture diagram for the Raspberry Pi heart sound classification system.
Style inspired by reference block diagram image.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.image import imread
import matplotlib.patheffects as pe

plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(0, 18)
ax.set_ylim(0, 8.6)
ax.axis('off')

# ── Colour palette ────────────────────────────────────────────────
C_BLE_HDR   = '#2980B9'   # blue  — BLE block header
C_BLE_BG    = '#D6EAF8'
C_PRE_HDR   = '#27AE60'   # green — preprocessing header
C_PRE_BG    = '#D5F5E3'
C_AI_HDR    = '#8E44AD'   # purple — AI inference header
C_AI_BG     = '#F5EEF8'
C_SYS_HDR   = '#E67E22'   # orange — system mgmt header
C_SYS_BG    = '#FEF9E7'
C_OUT_HDR   = '#2E86C1'   # blue — output header
C_OUT_BG    = '#EBF5FB'
C_PI_BDR    = '#E74C3C'   # red border — Pi enclosure
C_PI_BG     = '#FDEDEC'   # very light red
C_ESP_BG    = '#F8F9FA'
C_ESP_BDR   = '#555555'
C_ARROW     = '#444444'
C_ORANGE    = '#E67E22'


def rbox(ax, x, y, w, h, fc, ec, lw=1.5, pad=0.07, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))


def header_block(ax, x, y, w, h, title, bullets,
                 hdr_color, bg_color, ec_color,
                 title_fs=9, bullet_fs=8, zorder=3):
    """Draw a block with coloured header bar + bullet-point body."""
    hdr_h = 0.38
    # body
    rbox(ax, x, y, w, h, bg_color, ec_color, lw=1.5, zorder=zorder)
    # header bar
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h,
        boxstyle='round,pad=0.05',
        facecolor=hdr_color, edgecolor=ec_color,
        linewidth=1.5, zorder=zorder + 1))
    # header text
    ax.text(x + w/2, y + h - hdr_h/2, title,
            ha='center', va='center', fontsize=title_fs,
            fontweight='bold', color='white', zorder=zorder + 2)
    # bullets
    line_h = (h - hdr_h - 0.12) / max(len(bullets), 1)
    for i, b in enumerate(bullets):
        by = y + h - hdr_h - 0.18 - i * line_h
        ax.text(x + 0.15, by, f'• {b}',
                ha='left', va='top', fontsize=bullet_fs,
                color='#222222', zorder=zorder + 2)


def arrow_h(ax, x0, x1, y, color=C_ARROW, lw=1.8):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=12), zorder=10)


def arrow_v(ax, x, y0, y1, color=C_ARROW, lw=1.8):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=12), zorder=10)


def label(ax, x, y, s, fs=8, color='#333333', bold=False, ha='center'):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs,
            color=color, fontweight='bold' if bold else 'normal', zorder=11)


# ══════════════════════════════════════════════════════════════════
# 1. ESP32 block (left)
# ══════════════════════════════════════════════════════════════════
rbox(ax, 0.2, 2.8, 1.7, 2.6, C_ESP_BG, C_ESP_BDR, lw=2)
ax.text(1.05, 5.55, 'ESP32', ha='center', fontsize=12,
        fontweight='bold', color='#333')
ax.text(1.05, 5.1,  'Heart Sound\nSensor Node', ha='center',
        fontsize=8, color='#555', linespacing=1.4)
# sub-items
for i, t in enumerate(['INMP441(I\u00b2S)', 'I\u00b2S \u2192 2 kHz decimate', 'BLE broadcast']):
    ax.text(0.38, 5.0 - i * 0.62, f'• {t}',
            ha='left', va='top', fontsize=10, color='#444')

# BLE symbol text
ax.text(1.05, 2.5, 'BLE / PCM (int16)',
        ha='center', fontsize=10, color=C_BLE_HDR, style='italic')

# ══════════════════════════════════════════════════════════════════
# 2. Raspberry Pi enclosure
# ══════════════════════════════════════════════════════════════════
rbox(ax, 2.2, 0.3, 10.5, 7.3, C_PI_BG, C_PI_BDR, lw=2.5, pad=0.10, zorder=1)
ax.text(7.45, 8.00, 'Raspberry Pi 4B', ha='center', fontsize=10,
        fontweight='bold', color=C_PI_BDR, zorder=5)

# ── 2a. BLE Reception ────────────────────────────────────────────
header_block(ax, 2.35, 5.2, 2.2, 2.1,
             'BLE Reception',
             ['Scan & connect', 'Subscribe notify', 'Buffer assembly',
              'Chunk queue (20 s)'],
             C_BLE_HDR, C_BLE_BG, C_BLE_HDR,
             title_fs=11, bullet_fs=10)

# ── 2b. Preprocessing ────────────────────────────────────────────
header_block(ax, 2.35, 2.6, 2.2, 2.4,
             'Preprocessing',
             ['Bandpass filter\n  25–400 Hz',
              'Sliding window\n  2 s / 50% overlap',
              'Log-Mel spectrogram\n  64×64',
              'Peak normalisation'],
             C_PRE_HDR, C_PRE_BG, C_PRE_HDR,
             title_fs=11, bullet_fs=10)

# ── 2c. AI Inference — header only, body = embedded flowchart ────
header_block(ax, 4.8, 2.6, 5.0, 4.7,
             'AI Inference', [],          # no bullets — image fills body
             C_AI_HDR, C_AI_BG, C_AI_HDR,
             title_fs=11, bullet_fs=10)

# Embed dual-model flowchart image inside the body area
# Block: x [4.8, 9.8],  body y [2.6, 6.92]
_ai_img = imread('/home/agiuser/FypProj/photos/双模型.drawio.png')
ax.imshow(_ai_img, extent=[4.88, 9.72, 2.66, 6.88],
          aspect='auto', zorder=5)

# ── 2d. Storage ──────────────────────────────────────────────────
header_block(ax, 10.5, 4.85, 2.3, 2.45,
             'Storage',
             ['WAV file save', 'CSV inference log', 'JSON summary'],
             C_OUT_HDR, C_OUT_BG, C_OUT_HDR,
             title_fs=11, bullet_fs=10)

# ── 2e. Display ──────────────────────────────────────────────────
header_block(ax, 10.5, 2.6, 2.3, 2.0,
             'OLED Display',
             ['Result display', 'Sys-info display\n  (CPU/RAM/Temp)'],
             C_OUT_HDR, C_OUT_BG, C_OUT_HDR,
             title_fs=11, bullet_fs=10)

# ── 2f. System Management ────────────────────────────────────────
header_block(ax, 10.5, 0.55, 2.3, 1.85,
             'System Mgmt',
             ['Button UI\n  (start/stop/off)', 'Watchdog heartbeat'],
             C_SYS_HDR, C_SYS_BG, C_SYS_HDR,
             title_fs=11, bullet_fs=10)

# ══════════════════════════════════════════════════════════════════
# 3. Output column (right)
# ══════════════════════════════════════════════════════════════════
for yi, (title, color) in enumerate([
    ('Normal /\nAbnormal', '#1E8449'),
    ('WAV + CSV\nLog Files', '#1565C0'),
    ('OLED\nUI', '#7B1FA2'),
]):
    bx, by = 13.1, 5.6 - yi * 2.0
    rbox(ax, bx, by, 2.0, 1.4, '#FFFFFF', color, lw=2)
    ax.text(bx + 1.0, by + 0.7, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color=color,
            linespacing=1.4, zorder=5)

# ══════════════════════════════════════════════════════════════════
# 4. Arrows
# ══════════════════════════════════════════════════════════════════
# ESP32 → BLE reception
arrow_h(ax, 1.9, 2.35, 6.25, color=C_BLE_HDR)
label(ax, 2.12, 6.45, 'BLE', fs=7.5, color=C_BLE_HDR, bold=True)

# BLE reception → Preprocessing (vertical)
arrow_v(ax, 3.45, 5.2, 5.0, color=C_ARROW)

# Preprocessing → AI inference (horizontal)
arrow_h(ax, 4.55, 4.8, 4.95, color=C_ARROW)

# AI inference → Storage
arrow_h(ax, 9.8, 10.5, 6.5, color=C_ARROW)

# AI inference → Display
arrow_h(ax, 9.8, 10.5, 4.0, color=C_ARROW)

# Storage → output: Normal/Abnormal
arrow_h(ax, 12.8, 13.1, 6.3, color='#1E8449')

# Storage → output: Files
arrow_h(ax, 12.8, 13.1, 4.3, color='#1565C0')

# Display → output: OLED
arrow_h(ax, 12.8, 13.1, 3.3, color='#7B1FA2')

# ══════════════════════════════════════════════════════════════════
# 5. Title + legend
# ══════════════════════════════════════════════════════════════════
ax.set_title('System Architecture — Raspberry Pi Heart Sound Classification',
             fontsize=12, fontweight='bold', pad=12)

legend_handles = [
    mpatches.Patch(fc=C_BLE_BG,  ec=C_BLE_HDR,  label='BLE / Communication'),
    mpatches.Patch(fc=C_PRE_BG,  ec=C_PRE_HDR,  label='Signal Preprocessing'),
    mpatches.Patch(fc=C_AI_BG,   ec=C_AI_HDR,   label='AI Inference (TFLite INT8)'),
    mpatches.Patch(fc=C_OUT_BG,  ec=C_OUT_HDR,  label='Storage / Display'),
    mpatches.Patch(fc=C_SYS_BG,  ec=C_SYS_HDR,  label='System Management'),
    mpatches.Patch(fc=C_PI_BG,   ec=C_PI_BDR,   label='Raspberry Pi 4B boundary'),
]
ax.legend(handles=legend_handles, loc='lower left',
          bbox_to_anchor=(0.01, 0.01), fontsize=7.5,
          framealpha=0.9, ncol=2, edgecolor='#aaaaaa')

plt.tight_layout()
plt.savefig('fig_system_diagram.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print("fig_system_diagram.png 已保存")
