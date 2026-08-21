import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set figure size and resolution
fig, ax = plt.subplots(figsize=(12, 12), dpi=600)
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# Color palette matching user's reference image & pie chart theme exactly
c_plum = '#431C4A'       # Deep Plum / High Burnout (Main Vertical Stack)
c_rose = '#D6456E'       # Crimson Rose / Medium Burnout (Side Oval Badges)
c_coral = '#FA935A'      # Soft Coral / Low Burnout
c_navy = '#1B365D'       # Deep Navy (Header & Titles)
c_bg = '#FFFFFF'         # Clean White Background
c_border_gray = '#94A3B8'# Dotted frame border color

fig.patch.set_facecolor(c_bg)
ax.set_facecolor(c_bg)

# -------------------------------------------------------------------------
# TITLE HEADER
# -------------------------------------------------------------------------
plt.text(0.5, 11.4, "Figure 1: End-to-End Methodology & Machine Learning Architecture", 
         fontsize=16, fontweight='bold', color=c_navy, ha='left', va='center')
plt.text(0.5, 11.05, "Explanatory Sequential Mixed-Methods Architecture (QUAN → QUAL): Quantitative ML/XAI & Qualitative Triangulation", 
         fontsize=9.5, color='#475569', ha='left', va='center')

# Subtle divider line under header
ax.plot([0.5, 11.5], [10.85, 10.85], color='#E2E8F0', linewidth=1.5)

# -------------------------------------------------------------------------
# DOTTED FRAME AROUND SIDE DETAIL OVALS (Right column box like reference image)
# -------------------------------------------------------------------------
dotted_box = patches.FancyBboxPatch((7.0, 0.7), 4.5, 9.8,
                                    boxstyle="round,pad=0.1,rounding_size=0.1",
                                    facecolor='none', edgecolor=c_border_gray, 
                                    linestyle='--', linewidth=1.2, zorder=1)
ax.add_patch(dotted_box)

# -------------------------------------------------------------------------
# DEFINE MAIN STEPS (6 Vertical Plum Blocks + 6 Right Coral/Rose Ovals)
# -------------------------------------------------------------------------
steps = [
    {
        'num': '1',
        'title': 'DATA COLLECTION &\nPREPROCESSING',
        'detail': 'Google Form Survey (N=601)\nPrimary Survey Dataset',
        'icon': 'search',
        'y': 9.2
    },
    {
        'num': '2',
        'title': 'FEATURE ENGINEERING &\nPREPROCESSING',
        'detail': '9 Composite Feature Indices\nStandardScaler & OneHotEncoder',
        'icon': 'settings',
        'y': 7.5
    },
    {
        'num': '3',
        'title': 'MODEL TRAINING &\n10-FOLD CV',
        'detail': '10-Fold Stratified CV\nRandom Forest (65.89% Acc)',
        'icon': 'cpu',
        'y': 5.8
    },
    {
        'num': '4',
        'title': 'EXPLAINABLE AI (XAI) &\nSHAP ANALYSIS',
        'detail': 'SHAP Global & Local Importances\nConfusion Matrix Verification',
        'icon': 'chart',
        'y': 4.1
    },
    {
        'num': '5',
        'title': 'QUALITATIVE THEMATIC\nANALYSIS',
        'detail': 'Interviews (N=20 Students)\nBraun & Clarke 4 Core Themes',
        'icon': 'chat',
        'y': 2.4
    },
    {
        'num': '6',
        'title': 'TRIANGULATION &\nFINAL OUTPUT',
        'detail': 'Explanatory Triangulation Matrix\nInstitutional Early-Warning System',
        'icon': 'check',
        'y': 0.7
    }
]

box_w = 4.8
box_h = 1.1
oval_w = 4.0
oval_h = 0.95

# Helper function to draw vector icons inside left white circles
def draw_icon(ax, icon_name, cx, cy):
    if icon_name == 'search':
        circle = patches.Circle((cx-0.03, cy+0.03), 0.12, facecolor='none', edgecolor=c_plum, linewidth=2.0, zorder=5)
        ax.add_patch(circle)
        ax.plot([cx+0.05, cx+0.14], [cy-0.05, cy-0.14], color=c_plum, linewidth=2.2, zorder=5)
    elif icon_name == 'settings':
        ax.plot([cx-0.12, cx+0.12], [cy+0.08, cy+0.08], color=c_plum, linewidth=2.0, zorder=5)
        ax.plot([cx-0.12, cx+0.12], [cy-0.08, cy-0.08], color=c_plum, linewidth=2.0, zorder=5)
        c1 = patches.Circle((cx-0.03, cy+0.08), 0.04, facecolor=c_plum, edgecolor=c_plum, zorder=5)
        c2 = patches.Circle((cx+0.04, cy-0.08), 0.04, facecolor=c_plum, edgecolor=c_plum, zorder=5)
        ax.add_patch(c1); ax.add_patch(c2)
    elif icon_name == 'cpu':
        rect = patches.Rectangle((cx-0.1, cy-0.1), 0.2, 0.2, facecolor='none', edgecolor=c_plum, linewidth=2.0, zorder=5)
        ax.add_patch(rect)
        ax.plot([cx-0.16, cx-0.1], [cy, cy], color=c_plum, linewidth=1.8, zorder=5)
        ax.plot([cx+0.1, cx+0.16], [cy, cy], color=c_plum, linewidth=1.8, zorder=5)
        ax.plot([cx, cx], [cy-0.16, cy-0.1], color=c_plum, linewidth=1.8, zorder=5)
        ax.plot([cx, cx], [cy+0.1, cy+0.16], color=c_plum, linewidth=1.8, zorder=5)
    elif icon_name == 'chart':
        ax.plot([cx-0.12, cx+0.12], [cy-0.12, cy-0.12], color=c_plum, linewidth=2.0, zorder=5)
        ax.plot([cx-0.08, cx-0.08], [cy-0.12, cy], color=c_plum, linewidth=3.5, zorder=5)
        ax.plot([cx, cx], [cy-0.12, cy+0.1], color=c_plum, linewidth=3.5, zorder=5)
        ax.plot([cx+0.08, cx+0.08], [cy-0.12, cy+0.04], color=c_plum, linewidth=3.5, zorder=5)
    elif icon_name == 'chat':
        bubble = patches.Polygon([[cx-0.12, cy+0.1], [cx+0.12, cy+0.1], [cx+0.12, cy-0.06], [cx-0.02, cy-0.06], [cx-0.08, cy-0.14], [cx-0.06, cy-0.06], [cx-0.12, cy-0.06]], 
                                 closed=True, facecolor='none', edgecolor=c_plum, linewidth=2.0, zorder=5)
        ax.add_patch(bubble)
    elif icon_name == 'check':
        ax.plot([cx-0.12, cx-0.02, cx+0.12], [cy, cy-0.1, cy+0.12], color=c_plum, linewidth=2.5, zorder=5)

# -------------------------------------------------------------------------
# RENDER FLOWCHART ELEMENTS
# -------------------------------------------------------------------------
for i, step in enumerate(steps):
    y = step['y']
    x_box = 0.8
    x_oval = 7.25
    
    # 1. Main Plum Rectangle Box (Left Column)
    shadow = patches.FancyBboxPatch((x_box+0.05, y-0.05), box_w, box_h,
                                   boxstyle="round,pad=0.0,rounding_size=0.12",
                                   facecolor='#CBD5E1', edgecolor='none', zorder=1)
    ax.add_patch(shadow)
    
    plum_box = patches.FancyBboxPatch((x_box, y), box_w, box_h,
                                     boxstyle="round,pad=0.0,rounding_size=0.12",
                                     facecolor=c_plum, edgecolor='none', zorder=2)
    ax.add_patch(plum_box)
    
    # Left White Circle Container for Icon
    icon_circle = patches.Circle((x_box + 0.65, y + box_h/2), 0.38,
                                 facecolor='white', edgecolor='none', zorder=4)
    ax.add_patch(icon_circle)
    
    # Draw vector icon inside circle
    draw_icon(ax, step['icon'], x_box + 0.65, y + box_h/2)
    
    # White Bold Title Text inside Plum Box
    plt.text(x_box + 1.25, y + box_h/2, step['title'],
             color='white', fontsize=10.5, fontweight='bold', 
             va='center', ha='left', linespacing=1.2, zorder=4)
    
    # 2. Right Rose/Coral Detail Oval Badge (Alternating or Rose color match)
    oval_color = c_rose if i % 2 == 1 else c_coral
    
    oval_shadow = patches.Ellipse((x_oval + oval_w/2 + 0.04, y + box_h/2 - 0.04), oval_w, oval_h,
                                  facecolor='#CBD5E1', edgecolor='none', zorder=1)
    ax.add_patch(oval_shadow)
    
    coral_oval = patches.Ellipse((x_oval + oval_w/2, y + box_h/2), oval_w, oval_h,
                                 facecolor=oval_color, edgecolor='none', zorder=3)
    ax.add_patch(coral_oval)
    
    # Text inside Coral Oval
    plt.text(x_oval + oval_w/2, y + box_h/2, step['detail'],
             color='white', fontsize=8.8, fontweight='bold',
             va='center', ha='center', linespacing=1.3, zorder=4)
    
    # 3. Horizontal Arrow from Plum Box to Coral Oval
    arrow_h = patches.FancyArrowPatch((x_box + box_w + 0.05, y + box_h/2), 
                                      (x_oval + 0.05, y + box_h/2),
                                      connectionstyle="arc3,rad=0",
                                      arrowstyle="-|>",
                                      mutation_scale=14,
                                      linewidth=1.8,
                                      color=c_navy,
                                      zorder=4)
    ax.add_patch(arrow_h)
    
    # 4. Downward Chevron Arrow between Plum Boxes
    if i < len(steps) - 1:
        next_y = steps[i+1]['y']
        chevron = patches.Polygon([
            [x_box + box_w/2 - 0.25, y - 0.08],
            [x_box + box_w/2 + 0.25, y - 0.08],
            [x_box + box_w/2, next_y + box_h + 0.05]
        ], closed=True, facecolor=c_plum, edgecolor='none', zorder=4)
        ax.add_patch(chevron)

plt.tight_layout()
plt.savefig('Figure_1_Workflow.png', dpi=600, bbox_inches='tight')
plt.close()
print("Methodology flowchart regenerated and saved to Figure_1_Workflow.png!")
