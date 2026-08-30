import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from feature_engineering import prepare_features

print("Training full models and exporting JS-compatible inference data...")

# 1. Load Data
df = pd.read_excel('Quantitative_Survey_Data.xlsx')

# 2. Canonical Feature Engineering
df_full = prepare_features(df)
y_true = df_full['target'].values
X = df_full.drop(columns=['burnout_score', 'target']).copy()

cat_cols = ['gender', 'age_group', 'degree', 'academic_year']
num_cols = [c for c in X.columns if c not in cat_cols]

# 3. Fit Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
    ]
)

X_trans = preprocessor.fit_transform(X)

num_feature_names = num_cols
cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
all_feature_names = num_feature_names + cat_feature_names

# 4. Fit Random Forest
rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
rf.fit(X_trans, y_true)

# 5. Fit Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_trans, y_true)

# 6. Extract Preprocessor Parameters
scaler = preprocessor.named_transformers_['num']
ohe = preprocessor.named_transformers_['cat']

scaler_data = {
    'features': num_cols,
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}

ohe_categories = {}
for col, cats in zip(cat_cols, ohe.categories_):
    ohe_categories[col] = cats.tolist()

# 7. Extract Tree Structures for Random Forest
def export_tree(tree):
    return {
        'children_left': tree.children_left.tolist(),
        'children_right': tree.children_right.tolist(),
        'feature': tree.feature.tolist(),
        'threshold': tree.threshold.tolist(),
        'value': tree.value.tolist() # shape (n_nodes, 1, n_classes)
    }

rf_trees = [export_tree(estimator.tree_) for estimator in rf.estimators_]

# 8. Logistic Regression Weights
lr_data = {
    'intercept': lr.intercept_.tolist()[0],
    'coef': lr.coef_.tolist()[0]
}

# 9. Global Feature Importance from SHAP / Feature importances
rf_importances = {name: float(imp) for name, imp in zip(all_feature_names, rf.feature_importances_)}

# 10. Population Baseline Norms (Means and SDs for individual XAI deviation explanation)
population_norms = {}
for col in num_cols:
    population_norms[col] = {
        'mean': float(X[col].mean()),
        'std': float(X[col].std()),
        'median': float(X[col].median()),
        'min': float(X[col].min()),
        'max': float(X[col].max())
    }

# 11. Compile Export Object
export_bundle = {
    'feature_names': all_feature_names,
    'num_features': num_cols,
    'cat_features': cat_cols,
    'scaler': scaler_data,
    'ohe_categories': ohe_categories,
    'rf_trees': rf_trees,
    'lr_model': lr_data,
    'rf_feature_importances': rf_importances,
    'population_norms': population_norms,
    'metadata': {
        'total_samples': len(df),
        'high_burnout_rate': float(np.mean(y_true)),
        'rf_accuracy_10fold': 0.6589,
        'rf_roc_auc_10fold': 0.7126,
        'optimal_threshold': 0.38,
        'optimal_sensitivity': 0.7176,
        'optimal_specificity': 0.5607
    }
}

import os
os.makedirs('webapp/js', exist_ok=True)
with open('webapp/js/model_data.json', 'w', encoding='utf-8') as f:
    json.dump(export_bundle, f)

with open('webapp/js/model_data.js', 'w', encoding='utf-8') as f:
    f.write("const MODEL_DATA = " + json.dumps(export_bundle) + ";\nwindow.MODEL_DATA = MODEL_DATA;\n")

print(f"Successfully exported model data to webapp/js/model_data.json and webapp/js/model_data.js!")
print(f"Total features: {len(all_feature_names)} ({len(num_cols)} num, {len(cat_feature_names)} cat)")
print(f"Total RF Trees exported: {len(rf_trees)}")
