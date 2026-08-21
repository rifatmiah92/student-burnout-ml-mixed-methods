import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from feature_engineering import prepare_features

print("Running Genuine SHAP Feature Importance Analysis with Shared Feature Module...")

# 1. Load Data
df = pd.read_excel('Quantitative_Survey_Data.xlsx')

# 2. Shared Canonical Feature Engineering
df_full = prepare_features(df)
y_true = df_full['target'].values
X = df_full.drop(columns=['burnout_score', 'target']).copy()

cat_cols = ['gender', 'age_group', 'degree', 'academic_year']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
    ]
)

X_trans = preprocessor.fit_transform(X)

# Get feature names
num_feature_names = num_cols
cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
all_feature_names = num_feature_names + cat_feature_names

X_trans_df = pd.DataFrame(X_trans, columns=all_feature_names)

# Fit Random Forest
rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
rf.fit(X_trans_df, y_true)

# SHAP TreeExplainer
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X_trans_df)

if isinstance(shap_vals, list):
    vals = shap_vals[1]
elif len(shap_vals.shape) == 3:
    vals = shap_vals[:, :, 1]
else:
    vals = shap_vals

mean_abs_shap = np.mean(np.abs(vals), axis=0)

shap_summary_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Mean_SHAP_Value': mean_abs_shap
}).sort_values(by='Mean_SHAP_Value', ascending=False)

print("\n============================================================")
print("GLOBAL SHAP FEATURE IMPORTANCE (RANDOM FOREST)")
print("============================================================")
print(shap_summary_df.to_string(index=False))

# Save output
with open('results_shap_importance.txt', 'w', encoding='utf-8') as f:
    f.write("============================================================\n")
    f.write("GLOBAL SHAP FEATURE IMPORTANCE (RANDOM FOREST)\n")
    f.write("============================================================\n")
    f.write(shap_summary_df.to_string(index=False))

print("\nSHAP analysis completed successfully!")
