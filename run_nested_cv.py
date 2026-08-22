import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from feature_engineering import prepare_features

print("Running Genuine 10-Fold Outer / 5-Fold Inner Nested Cross-Validation Tuning...")

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
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
    ]
)

outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Parameter grids for tuning
param_grids = {
    'Random Forest': {
        'clf__n_estimators': [100, 150],
        'clf__max_depth': [6, 8, 10],
        'clf__min_samples_split': [2, 5]
    },
    'Logistic Regression': {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__penalty': ['l2']
    }
}

base_classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

nested_results = []

for model_name, clf in base_classifiers.items():
    start_t = time.time()
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', clf)
    ])
    
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[model_name],
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=1
    )
    
    oof_preds = np.zeros(len(y_true))
    oof_probas = np.zeros(len(y_true))
    
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_true)):
        X_train, y_train = X.iloc[train_idx], y_true[train_idx]
        X_val, y_val = X.iloc[val_idx], y_true[val_idx]
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        oof_preds[val_idx] = best_model.predict(X_val)
        oof_probas[val_idx] = best_model.predict_proba(X_val)[:, 1]
        
    elapsed = round(time.time() - start_t, 2)
    
    acc = round(accuracy_score(y_true, oof_preds), 4)
    prec = round(precision_score(y_true, oof_preds), 4)
    rec = round(recall_score(y_true, oof_preds), 4)
    f1 = round(f1_score(y_true, oof_preds), 4)
    auc = round(roc_auc_score(y_true, oof_probas), 4)
    
    nested_results.append({
        'Model': f"{model_name} (Nested CV Tuned)",
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC-AUC': auc,
        'Time (s)': elapsed
    })
    
    print(f"Completed {model_name} (Nested CV Tuned) - Acc: {acc*100:.2f}%, AUC: {auc:.4f} in {elapsed}s")

res_df = pd.DataFrame(nested_results)

print("\n=========================================================================")
print("NESTED CROSS-VALIDATION HYPERPARAMETER TUNING RESULTS")
print("=========================================================================")
print(res_df.to_string(index=False))

with open('results_nested_cv.txt', 'w', encoding='utf-8') as f:
    f.write("=========================================================================\n")
    f.write("NESTED CROSS-VALIDATION HYPERPARAMETER TUNING RESULTS\n")
    f.write("=========================================================================\n")
    f.write(res_df.to_string(index=False) + "\n")

print("\nNested CV tuning completed successfully!")
