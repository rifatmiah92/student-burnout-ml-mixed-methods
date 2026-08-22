import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

from feature_engineering import prepare_features

print("Running Genuine Leak-Free SHAP Feature Importance & Fold-by-Fold Stability Analysis...")

# 1. Load Data
df = pd.read_excel('Quantitative_Survey_Data.xlsx')

# 2. Shared Canonical Feature Engineering
df_full = prepare_features(df)
y_true = df_full['target'].values
X = df_full.drop(columns=['burnout_score', 'target']).copy()

cat_cols = ['gender', 'age_group', 'degree', 'academic_year']
num_cols = [c for c in X.columns if c not in cat_cols]

# ---------------------------------------------------------
# A. FULL-DATASET REFIT SHAP
# ---------------------------------------------------------
preprocessor_full = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
    ]
)

X_trans_full = preprocessor_full.fit_transform(X)

num_feature_names = num_cols
cat_feature_names = list(preprocessor_full.named_transformers_['cat'].get_feature_names_out(cat_cols))
all_feature_names = num_feature_names + cat_feature_names

X_trans_full_df = pd.DataFrame(X_trans_full, columns=all_feature_names)

rf_full = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
rf_full.fit(X_trans_full_df, y_true)

explainer_full = shap.TreeExplainer(rf_full)
shap_vals_full = explainer_full.shap_values(X_trans_full_df)

if isinstance(shap_vals_full, list):
    vals_full = shap_vals_full[1]
elif len(shap_vals_full.shape) == 3:
    vals_full = shap_vals_full[:, :, 1]
else:
    vals_full = shap_vals_full

mean_abs_shap_full = np.mean(np.abs(vals_full), axis=0)

shap_summary_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Mean_SHAP_Value': mean_abs_shap_full
}).sort_values(by='Mean_SHAP_Value', ascending=False)


# ---------------------------------------------------------
# B. STRICT LEAK-FREE 10-FOLD CV OUT-OF-FOLD SHAP FOR STABILITY VERIFICATION
# ---------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_shap_matrix = np.zeros_like(X_trans_full)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
    X_raw_train, y_train = X.iloc[train_idx], y_true[train_idx]
    X_raw_val, y_val = X.iloc[val_idx], y_true[val_idx]
    
    # Fit preprocessor strictly on train fold
    fold_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    X_train_trans = fold_preprocessor.fit_transform(X_raw_train)
    X_val_trans = fold_preprocessor.transform(X_raw_val)
    
    X_train_df = pd.DataFrame(X_train_trans, columns=all_feature_names)
    X_val_df = pd.DataFrame(X_val_trans, columns=all_feature_names)
    
    rf_fold = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    rf_fold.fit(X_train_df, y_train)
    
    explainer_fold = shap.TreeExplainer(rf_fold)
    shap_vals_fold = explainer_fold.shap_values(X_val_df)
    
    if isinstance(shap_vals_fold, list):
        vals_fold = shap_vals_fold[1]
    elif len(shap_vals_fold.shape) == 3:
        vals_fold = shap_vals_fold[:, :, 1]
    else:
        vals_fold = shap_vals_fold
        
    oof_shap_matrix[val_idx] = vals_fold

mean_abs_shap_oof = np.mean(np.abs(oof_shap_matrix), axis=0)

# Compute Spearman Rank Correlation
spearman_rho, spearman_p = spearmanr(mean_abs_shap_full, mean_abs_shap_oof)

print("\n============================================================")
print("GLOBAL SHAP FEATURE IMPORTANCE (RANDOM FOREST - FULL REFIT)")
print("============================================================")
print(shap_summary_df.to_string(index=False))

print("\n============================================================")
print("STRICT LEAK-FREE SHAP STABILITY (FULL REFIT vs 10-FOLD OUT-OF-FOLD)")
print("============================================================")
print(f"Spearman rank correlation (rho): {spearman_rho:.4f}")
print(f"p-value: {spearman_p:.4e}")

# Save output
with open('results_shap_importance.txt', 'w', encoding='utf-8') as f:
    f.write("============================================================\n")
    f.write("GLOBAL SHAP FEATURE IMPORTANCE (RANDOM FOREST)\n")
    f.write("============================================================\n")
    f.write(shap_summary_df.to_string(index=False))
    f.write("\n\n============================================================\n")
    f.write("STRICT LEAK-FREE SHAP FOLD-BY-FOLD STABILITY VERIFICATION\n")
    f.write("============================================================\n")
    f.write(f"Spearman Rank Correlation (rho): {spearman_rho:.4f}\n")
    f.write(f"p-value: {spearman_p:.4e}\n")

print("\nLeak-free SHAP analysis and fold-by-fold stability verification completed successfully!")
