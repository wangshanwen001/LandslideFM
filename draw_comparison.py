import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
fig.patch.set_facecolor('white')

for ax in [ax1, ax2]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

C_DATA = '#E8F4FD'
C_TRAD_BACKBONE = '#FFE0E0'
C_TRAD_HEAD = '#FFC0C0'
C_OURS_BACKBONE = '#D4EDDA'
C_SEMI = '#C8E6C9'
C_HEAD_SEG = '#B3E5FC'
C_HEAD_CLS = '#FFE0B2'
C_HEAD_DET = '#E1BEE7'
C_VLM = '#FFF9C4'
C_ARROW = '#555555'
C_BORDER = '#333333'

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False, text_color='#222'):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=C_BORDER, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

def draw_dashed_arrow(ax, x1, y1, x2, y2, color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle='--'))

ax1.set_title('(a) Traditional Segmentation Pipeline', fontsize=14, fontweight='bold', pad=15)

# Data
draw_box(ax1, 2.5, 10.5, 5, 1, 'Remote Sensing Dataset', C_DATA, fontsize=11, bold=True)
draw_arrow(ax1, 5, 10.5, 5, 9.8)

# Supervised training box
draw_box(ax1, 2, 8.6, 6, 1.2, 'Supervised Training\n(Labeled Data Only)', C_TRAD_BACKBONE, fontsize=10, bold=True)
draw_arrow(ax1, 5, 8.6, 5, 7.9)

# Backbone + head (tied together)
draw_box(ax1, 1.5, 5.5, 7, 2.4, '', 'white', fontsize=10)
# backbone
draw_box(ax1, 1.8, 6.3, 3, 1.3, 'Backbone\n(CNN / ViT)\ne.g. ResNet, ViT', C_TRAD_BACKBONE, fontsize=9, bold=True)
# head
draw_box(ax1, 5.2, 6.3, 3, 1.3, 'Segmentation\nHead', C_TRAD_HEAD, bold=True)
# connector line
ax1.plot([4.8, 5.2], [6.95, 6.95], color=C_BORDER, linewidth=2)
# bounding box
border = FancyBboxPatch((1.5, 5.5), 7, 2.4, boxstyle="round,pad=0.1",
                          facecolor='none', edgecolor=C_BORDER, linewidth=1.2, linestyle='--')
ax1.add_patch(border)
ax1.text(5, 5.7, 'Trained Model (Task-Specific)', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#555', style='italic')

draw_arrow(ax1, 5, 5.5, 5, 4.8)

# Single output
draw_box(ax1, 3, 3.5, 4, 1.3, 'Segmentation Mask\n(Single Task)', C_TRAD_HEAD, bold=True)

# Limitation annotations
ax1.text(5, 2.3, 'X  Only one task\nX  No semantic understanding\nX  Limited to supervised data',
         ha='center', va='center', fontsize=10, color='#C62828', fontweight='bold')

ax2.set_title('(b) Ours: UFFM + DINOv2/DPT Unified Understanding Framework', fontsize=14, fontweight='bold', pad=15)

# Data
draw_box(ax2, 1, 10.5, 3.5, 1, 'Labeled Data', C_DATA, fontsize=10, bold=True)
draw_box(ax2, 5.5, 10.5, 3.5, 1, 'Unlabeled Data', C_DATA, fontsize=10, bold=True)
draw_arrow(ax2, 2.75, 10.5, 3.5, 9.8)
draw_arrow(ax2, 7.25, 10.5, 6.5, 9.8)

# Semi-supervised training
draw_box(ax2, 1.5, 8.6, 7, 1.2, 'UFFM Semi-Supervised Pre-training\n(FixMatch + SAM Pseudo-Labels + Feature Memory Bank)', C_SEMI, fontsize=9, bold=True)
draw_arrow(ax2, 5, 8.6, 5, 7.9)

# Shared backbone
draw_box(ax2, 2.5, 6.5, 5, 1.4, 'Shared Backbone\n(DINOv2 + DPT)', C_OURS_BACKBONE, fontsize=11, bold=True)
draw_arrow(ax2, 5, 6.5, 5, 5.8)

# Multiple heads row
# Seg head
draw_box(ax2, 0.3, 4.2, 2.8, 1.4, 'Segmentation\nHead', C_HEAD_SEG, bold=True)
# Cls head
draw_box(ax2, 3.6, 4.2, 2.8, 1.4, 'Classification\nHead', C_HEAD_CLS, bold=True)
# Det head
draw_box(ax2, 6.9, 4.2, 2.8, 1.4, 'Detection\nHead', C_HEAD_DET, bold=True)

# arrows from backbone to heads
draw_arrow(ax2, 3.5, 5.8, 1.7, 5.6)
draw_arrow(ax2, 5, 5.8, 5, 5.6)
draw_arrow(ax2, 6.5, 5.8, 8.3, 5.6)

# VLM understanding branch (from seg head)
draw_box(ax2, 2.5, 1.5, 5, 2, 'VLM Understanding Module\n(Qwen2.5-VL)\n\nEnvironmental Analysis | Impact Assessment | Visual QA',
         C_VLM, fontsize=9, bold=True)
draw_dashed_arrow(ax2, 1.7, 4.2, 5, 3.5, '#F9A825')

# Outputs row
# Seg output
draw_box(ax2, 0.3, 2.8, 2.8, 0.9, 'Seg Mask', C_HEAD_SEG, fontsize=9)
# Cls output
draw_box(ax2, 3.6, 2.8, 2.8, 0.9, 'Class Labels', C_HEAD_CLS, fontsize=9)
# Det output
draw_box(ax2, 6.9, 2.8, 2.8, 0.9, 'BBoxes', C_HEAD_DET, fontsize=9)

draw_arrow(ax2, 1.7, 4.2, 1.7, 3.7)
draw_arrow(ax2, 5, 4.2, 5, 3.7)
draw_arrow(ax2, 8.3, 4.2, 8.3, 3.7)

# Advantage annotations
ax2.text(5, 0.5, 'O  Multi-task adaptation  O  Semi-supervised efficiency  O  Semantic understanding via VLM',
         ha='center', va='center', fontsize=10, color='#2E7D32', fontweight='bold')

legend_elements = [
    mpatches.Patch(facecolor=C_DATA, edgecolor=C_BORDER, label='Data'),
    mpatches.Patch(facecolor=C_OURS_BACKBONE, edgecolor=C_BORDER, label='Backbone'),
    mpatches.Patch(facecolor=C_SEMI, edgecolor=C_BORDER, label='Semi-supervised Training'),
    mpatches.Patch(facecolor=C_HEAD_SEG, edgecolor=C_BORDER, label='Task Head'),
    mpatches.Patch(facecolor=C_VLM, edgecolor=C_BORDER, label='VLM Module'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=10, frameon=True, fancybox=True, shadow=False)

plt.tight_layout(rect=[0, 0.08, 1, 0.98])
plt.savefig('pipeline_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved to pipeline_comparison.png")
plt.savefig('pipeline_comparison.pdf', bbox_inches='tight', facecolor='white')
print("✓ Saved to pipeline_comparison.pdf")
