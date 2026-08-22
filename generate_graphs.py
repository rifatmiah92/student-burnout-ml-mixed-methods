import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap

# Signature Theme Color Palette
c_plum = '#481D4A'   # High Burnout / Deep Dark Plum
c_rose = '#D6456E'   # Medium Burnout / Crimson Rose Red
c_coral = '#FA935A'  # Low Burnout / Soft Coral Orange
c_navy = '#1B365D'   # Title & Axes Text (Deep Navy)

# Set global matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = c_navy
plt.rcParams['axes.linewidth'] = 1.4

# Load Dataset
data_path = "Quantitative_Survey_Data.xlsx"
df = pd.read_excel(data_path)

def categorize_burnout(score):
    if score == 3: return 'High Burnout'
    elif score == 2: return 'Medium Burnout'
    else: return 'Low Burnout'

df['burnout_level'] = df['burnout_score'].apply(categorize_burnout)

# =========================================================================
# FIGURE 2: Distribution of Student Burnout Levels (Pie/Donut Chart)
# =========================================================================
plt.figure(figsize=(9, 6.5), dpi=600)
counts = [
    (df['burnout_level'] == 'High Burnout').sum(),
    (df['burnout_level'] == 'Medium Burnout').sum(),
    (df['burnout_level'] == 'Low Burnout').sum()
]

colors_pie = [c_plum, c_rose, c_coral]
labels = [f"High Burnout ({counts[0]} students)", 
          f"Medium Burnout ({counts[1]} students)", 
          f"Low Burnout ({counts[2]} students)"]

wedges, texts, autotexts = plt.pie(
    counts, 
    colors=colors_pie, 
    autopct='%1.1f%%', 
    startangle=140, 
    pctdistance=0.75,
    textprops=dict(color="white", weight="bold", fontsize=13)
)

centre_circle = plt.Circle((0,0), 0.52, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Figure 2: Distribution of Student Burnout Levels', fontweight='bold', color=c_navy, fontsize=16, pad=20, loc='left')
plt.legend(wedges, labels, title="Burnout Levels", loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=True)
plt.tight_layout()
plt.savefig('Figure_2_Distribution.png', dpi=600, bbox_inches='tight')
plt.close()

# =========================================================================
# FIGURE 3: Burnout Levels across Gender
# =========================================================================
fig, ax_g = plt.subplots(figsize=(9, 5.8), dpi=600)
fig.patch.set_facecolor('#FFFFFF')
ax_g.set_facecolor('#FFFFFF')

sns.countplot(
    data=df, 
    x='gender', 
    hue='burnout_level', 
    hue_order=['Low Burnout', 'Medium Burnout', 'High Burnout'],
    palette=[c_coral, c_rose, c_plum], 
    ax=ax_g,
    linewidth=0
)

plt.title('Figure 3: Burnout Levels across Gender', fontweight='bold', color=c_navy, fontsize=15, pad=20)
ax_g.set_xlabel('Gender', fontweight='bold', color=c_navy, fontsize=12, labelpad=12)
ax_g.set_ylabel('Number of Students', fontweight='bold', color=c_navy, fontsize=12, labelpad=12)

ax_g.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
ax_g.set_yticklabels(['0', '20', '40', '60', '80', '100', '120', '140'], color=c_navy, fontsize=11)

plt.legend(title='Burnout Level', frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', loc='upper left')

sns.despine(top=True, right=True)
ax_g.spines['left'].set_color(c_navy)
ax_g.spines['left'].set_linewidth(2.0)
ax_g.spines['bottom'].set_color(c_navy)
ax_g.spines['bottom'].set_linewidth(2.0)

ax_g.grid(axis='y', linestyle='--', alpha=0.5, color='#E2E8F0')
ax_g.grid(axis='x', visible=False)

plt.tight_layout()
plt.savefig('Figure_3_Gender.png', dpi=600, bbox_inches='tight')
plt.close()

# =========================================================================
# FIGURE 4: Machine Learning Model Accuracies under Genuine 10-Fold CV
# =========================================================================
res_file = 'results_ml_summary.txt'
models = []
accuracies = []

if not os.path.exists(res_file):
    import subprocess
    print("results_ml_summary.txt not found. Running train_ml.py...")
    subprocess.run(['python', 'train_ml.py'], check=True)

with open(res_file, 'r', encoding='utf-8') as f:
    lines_res = f.readlines()

for line in lines_res[1:]: # Skip header
    line = line.strip()
    if not line or line.startswith('=') or line.startswith('MCNEMAR') or line.startswith('Pair'):
        continue
    parts = line.rsplit(maxsplit=6)
    if len(parts) == 7:
        try:
            m_name = parts[0]
            m_acc = float(parts[1])
            models.append(m_name)
            accuracies.append(m_acc)
        except ValueError:
            pass

bar_colors_f2 = []
for m, acc in zip(models, accuracies):
    if m == 'Random Forest' or m == 'Soft Voting Ensemble':
        bar_colors_f2.append(c_plum)  # Top Champion Models (65.89%)
    elif acc >= 0.63:
        bar_colors_f2.append(c_rose)  # Top Tier
    else:
        bar_colors_f2.append(c_coral) # Baseline Tier

fig, ax_m = plt.subplots(figsize=(11, 7), dpi=600)
fig.patch.set_facecolor('#FFFFFF')
ax_m.set_facecolor('#FFFFFF')

y_pos_m = np.arange(len(models))
bars_m = ax_m.barh(y_pos_m, accuracies[::-1], color=bar_colors_f2[::-1], edgecolor=c_navy, height=0.65, linewidth=1.2, zorder=3)

plt.title("Figure 4: Machine Learning Model Accuracies under 10-Fold Stratified CV", fontsize=15, fontweight='bold', color=c_navy, pad=15)
ax_m.set_xlabel('10-Fold Stratified Cross-Validation Accuracy Score', fontweight='bold', color=c_navy, fontsize=12, labelpad=10)
ax_m.set_ylabel('Algorithm / Ensemble Architecture', fontweight='bold', color=c_navy, fontsize=12, labelpad=10)
ax_m.set_xlim(0.50, 0.70)

ax_m.set_xticks([0.50, 0.55, 0.5757, 0.60, 0.65, 0.70])
ax_m.set_xticklabels(['50.0%', '55.0%', '57.6%\n(Baseline)', '60.0%', '65.0%', '70.0%'], fontweight='bold', color=c_navy)

# Draw baseline line at 57.57%
ax_m.axvline(x=0.5757, color='#DC2626', linestyle='--', linewidth=1.5, label='Majority Class Baseline (57.57%)', zorder=4)

ax_m.set_yticks(y_pos_m)
ax_m.set_yticklabels(models[::-1], fontweight='bold', color=c_navy)

acc_reversed_m = accuracies[::-1]
for bar, acc in zip(bars_m, acc_reversed_m):
    width = bar.get_width()
    txt = f" {acc*100:.2f}%"
    txt_color = c_plum if acc >= 0.64 else (c_navy if acc >= 0.62 else '#334155')
    ax_m.text(width + 0.003, bar.get_y() + bar.get_height()/2.0, txt,
            ha='left', va='center', fontsize=10.5, fontweight='bold', color=txt_color, zorder=5)

plt.legend(loc='lower right', frameon=True)
sns.despine(top=True, right=True)
ax_m.grid(axis='x', linestyle='--', alpha=0.5, color='#94A3B8', zorder=0)

plt.subplots_adjust(left=0.28, right=0.94, top=0.92, bottom=0.10)
plt.savefig('Figure_4_ML_Accuracies.png', dpi=600, bbox_inches='tight')
plt.close()


# =========================================================================
# FIGURE 5: Out-of-Fold Confusion Matrix Heatmap (Random Forest)
# =========================================================================
plt.figure(figsize=(6.5, 5.5), dpi=600)
cm_file = 'results_confusion_matrix.txt'
if os.path.exists(cm_file):
    cm = np.loadtxt(cm_file, dtype=int)
else:
    raise FileNotFoundError("results_confusion_matrix.txt not found! Run train_ml.py first.")

cmap_cm = LinearSegmentedColormap.from_list("cm_theme", ["#FFFFFF", c_coral, c_rose, c_plum])

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap=cmap_cm, 
    cbar=True, 
    annot_kws={"size": 16, "weight": "bold"},
    xticklabels=['Predicted Low/Med', 'Predicted High'], 
    yticklabels=['Actual Low/Med', 'Actual High'],
    linewidths=1.2, 
    linecolor=c_navy
)

plt.title('Figure 5: Confusion Matrix (Random Forest - 65.89% CV Accuracy)', fontweight='bold', color=c_navy, pad=15)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('Figure_5_Confusion_Matrix.png', dpi=600, bbox_inches='tight')
plt.close()

# =========================================================================
# FIGURE 6: Global SHAP Feature Importances (Random Forest Model)
# =========================================================================
shap_file = 'results_shap_importance.txt'
features_raw, shap_values_list = [], []

if os.path.exists(shap_file):
    with open(shap_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('Feature') or line.startswith('GLOBAL'):
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                try:
                    val = float(parts[1])
                    features_raw.append(parts[0])
                    shap_values_list.append(val)
                except ValueError:
                    pass

shap_df = pd.DataFrame({
    'Feature': features_raw,
    'Mean_SHAP_Value': shap_values_list
}).sort_values(by='Mean_SHAP_Value', ascending=False).head(10)

FEATURE_NAME_MAP = {
    'academic_performance_index': 'Academic Performance Index',
    'cgpa_midpoint': 'CGPA (Midpoint)',
    'screen_to_sleep_ratio': 'Screen-to-Sleep Ratio',
    'burnout_vulnerability_index': 'Burnout Vulnerability Index',
    'social_media_hours': 'Social Media Hours / Day',
    'study_to_rest_ratio': 'Study-to-Rest Ratio',
    'psychological_strain_index': 'Psychological Strain Index',
    'motivation_deficit_score': 'Motivation Deficit Score',
    'depression_score': 'Depression Score',
    'stress_score': 'Perceived Stress Score',
    'sleep_deprivation_index': 'Sleep Deprivation Index',
    'attendance_pct': 'Attendance Percentage',
    'part_time_score': 'Part-Time Workload Score',
    'motivation_score': 'Motivation Score',
    'study_hours_numeric': 'Study Hours / Day',
    'academic_pressure_index': 'Academic Pressure Index',
    'academic_pressure_score': 'Academic Pressure Score',
    'sleep_hours_numeric': 'Sleep Duration (Hours)',
    'sleep_quality_score': 'Sleep Quality Score',
    'physical_activity_hours': 'Physical Activity Hours',
    'wellbeing_buffer': 'Wellbeing Buffer Index'
}

features = [FEATURE_NAME_MAP.get(f, f.replace('_', ' ').title()) for f in shap_df['Feature']]
shap_values = shap_df['Mean_SHAP_Value'].tolist()

cmap_bar = LinearSegmentedColormap.from_list("custom_theme", [c_plum, c_rose, c_coral])
shap_colors = [cmap_bar(i / max(1, len(features) - 1)) for i in range(len(features))]

fig, ax_s = plt.subplots(figsize=(11, 6.5), dpi=600)
fig.patch.set_facecolor('#FFFFFF')
ax_s.set_facecolor('#FFFFFF')

y_pos = np.arange(len(features))
bars = ax_s.barh(y_pos, shap_values[::-1], color=shap_colors[::-1], edgecolor=c_navy, height=0.68, linewidth=1.2, zorder=3)

plt.title('Figure 6: Global SHAP Feature Importances (Random Forest Model)', fontweight='bold', color=c_navy, fontsize=15, pad=15)
ax_s.set_xlabel('Mean |SHAP Value| (Average Marginal Contribution to Burnout Risk)', fontweight='bold', color=c_navy, fontsize=12, labelpad=10)
ax_s.set_ylabel('Predictive Feature / Domain Index', fontweight='bold', color=c_navy, fontsize=12, labelpad=10)
ax_s.set_xlim(0, 0.045)

ax_s.set_yticks(y_pos)
ax_s.set_yticklabels(features[::-1], fontweight='bold', color=c_navy)

shap_reversed = shap_values[::-1]
for bar, val in zip(bars, shap_reversed):
    width = bar.get_width()
    ax_s.text(width + 0.0008, bar.get_y() + bar.get_height()/2.0, f"{val:.4f}",
            ha='left', va='center', fontsize=11, fontweight='bold', color=c_navy, zorder=5)

sns.despine(top=True, right=True)
ax_s.grid(axis='x', linestyle='--', alpha=0.5, color='#94A3B8', zorder=0)

plt.tight_layout()
plt.savefig('Figure_6_SHAP.png', dpi=600, bbox_inches='tight')
plt.close()

print("All publication figures (Figure 2, 3, 4, 5, 6) regenerated successfully!")

