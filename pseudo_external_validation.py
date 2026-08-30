import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from feature_engineering import prepare_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

print("Starting Pseudo-External Subgroup Validation Script...")

# 1. Load Data & Prepare Canonical Features
df = pd.read_excel('Quantitative_Survey_Data.xlsx')
df_full = prepare_features(df)

# 2. Partition into Subgroups by Academic Degree Level
bach_mask = df_full['degree'].str.contains("Bachelor", case=False, na=False)
grad_mask = ~bach_mask

df_bach = df_full[bach_mask].copy()
df_grad = df_full[grad_mask].copy()

print(f"Subgroup counts:")
print(f"  - Bachelor's cohort (Train): N = {len(df_bach)} (High burnout = {df_bach['target'].sum()}, {df_bach['target'].mean()*100:.1f}%)")
print(f"  - Master's/PhD/Diploma cohort (Test): N = {len(df_grad)} (High burnout = {df_grad['target'].sum()}, {df_grad['target'].mean()*100:.1f}%)")

# Exclude 'degree' from predictors as it defines the split
cat_cols = ['gender', 'age_group', 'academic_year']
num_cols = [c for c in df_full.drop(columns=['burnout_score', 'target', 'degree']).columns if c not in cat_cols]

def evaluate_subgroup_split(train_df, test_df, split_name):
    X_train = train_df.drop(columns=['burnout_score', 'target', 'degree']).copy()
    y_train = train_df['target'].values
    
    X_test = test_df.drop(columns=['burnout_score', 'target', 'degree']).copy()
    y_test = test_df['target'].values
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42))
    ])
    
    # Internal 10-fold CV on training subgroup
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_cv_pred = cross_val_predict(pipe, X_train, y_train, cv=cv, method='predict')
    y_cv_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    cv_acc = round(accuracy_score(y_train, y_cv_pred) * 100, 2)
    cv_auc = round(roc_auc_score(y_train, y_cv_proba), 4)
    cv_f1 = round(f1_score(y_train, y_cv_pred), 4)
    
    # Train on full training subgroup and evaluate on held-out test subgroup
    pipe.fit(X_train, y_train)
    y_test_pred = pipe.predict(X_test)
    y_test_proba = pipe.predict_proba(X_test)[:, 1]
    
    test_acc = round(accuracy_score(y_test, y_test_pred) * 100, 2)
    test_prec = round(precision_score(y_test, y_test_pred) * 100, 2)
    test_rec = round(recall_score(y_test, y_test_pred) * 100, 2)
    test_f1 = round(f1_score(y_test, y_test_pred), 4)
    test_auc = round(roc_auc_score(y_test, y_test_proba), 4)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    return {
        'Split Name': split_name,
        'Train N': len(train_df),
        'Test N': len(test_df),
        'Internal CV Acc (%)': cv_acc,
        'Internal CV AUC': cv_auc,
        'Internal CV F1': cv_f1,
        'Held-Out Acc (%)': test_acc,
        'Held-Out Prec (%)': test_prec,
        'Held-Out Rec (%)': test_rec,
        'Held-Out F1': test_f1,
        'Held-Out AUC': test_auc,
        'TN': test_cm[0, 0], 'FP': test_cm[0, 1],
        'FN': test_cm[1, 0], 'TP': test_cm[1, 1]
    }

res_forward = evaluate_subgroup_split(df_bach, df_grad, "Forward: Bachelor's -> Master's/PhD/Diploma")
res_reverse = evaluate_subgroup_split(df_grad, df_bach, "Reverse: Master's/PhD/Diploma -> Bachelor's")

summary_lines = []
summary_lines.append("=====================================================================================")
summary_lines.append("PSEUDO-EXTERNAL SUBGROUP VALIDATION RESULTS (RANDOM FOREST)")
summary_lines.append("=====================================================================================")
summary_lines.append(f"\n1. Primary Evaluation: Bachelor's (Train N={res_forward['Train N']}) -> Master's/PhD/Diploma (Test N={res_forward['Test N']})")
summary_lines.append(f"   - Internal 10-Fold CV Accuracy: {res_forward['Internal CV Acc (%)']:.2f}%, ROC-AUC: {res_forward['Internal CV AUC']:.4f}, F1: {res_forward['Internal CV F1']:.4f}")
summary_lines.append(f"   - Held-Out Subgroup Accuracy: {res_forward['Held-Out Acc (%)']:.2f}%")
summary_lines.append(f"   - Held-Out Precision: {res_forward['Held-Out Prec (%)']:.2f}%")
summary_lines.append(f"   - Held-Out Recall (Sensitivity): {res_forward['Held-Out Rec (%)']:.2f}%")
summary_lines.append(f"   - Held-Out F1 Score: {res_forward['Held-Out F1']:.4f}")
summary_lines.append(f"   - Held-Out ROC-AUC: {res_forward['Held-Out AUC']:.4f}")
summary_lines.append(f"   - Confusion Matrix [TN, FP / FN, TP]: [{res_forward['TN']}, {res_forward['FP']} / {res_forward['FN']}, {res_forward['TP']}]")

summary_lines.append(f"\n2. Sensitivity Check: Master's/PhD/Diploma (Train N={res_reverse['Train N']}) -> Bachelor's (Test N={res_reverse['Test N']})")
summary_lines.append(f"   - Internal 10-Fold CV Accuracy: {res_reverse['Internal CV Acc (%)']:.2f}%, ROC-AUC: {res_reverse['Internal CV AUC']:.4f}, F1: {res_reverse['Internal CV F1']:.4f}")
summary_lines.append(f"   - Held-Out Subgroup Accuracy: {res_reverse['Held-Out Acc (%)']:.2f}%")
summary_lines.append(f"   - Held-Out Precision: {res_reverse['Held-Out Prec (%)']:.2f}%")
summary_lines.append(f"   - Held-Out Recall (Sensitivity): {res_reverse['Held-Out Rec (%)']:.2f}%")
summary_lines.append(f"   - Held-Out F1 Score: {res_reverse['Held-Out F1']:.4f}")
summary_lines.append(f"   - Held-Out ROC-AUC: {res_reverse['Held-Out AUC']:.4f}")
summary_lines.append(f"   - Confusion Matrix [TN, FP / FN, TP]: [{res_reverse['TN']}, {res_reverse['FP']} / {res_reverse['FN']}, {res_reverse['TP']}]")

summary_text = "\n".join(summary_lines)
print(summary_text)

with open('results_pseudo_external.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text + "\n")

print("\nResults saved to results_pseudo_external.txt successfully!")
