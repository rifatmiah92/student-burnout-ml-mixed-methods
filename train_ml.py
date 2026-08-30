import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from feature_engineering import prepare_features

print("Starting REAL Scikit-Learn Model Training for ALL 10 Models & Ensemble...")

# 1. Load Data
df = pd.read_excel('Quantitative_Survey_Data.xlsx')

# 2. Canonical Feature Engineering (Shared Module)
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

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'CatBoost Classifier': CatBoostClassifier(iterations=150, learning_rate=0.08, verbose=0, random_state=42),
    'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42),
    'Support Vector Machine (SVM)': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
    'Multilayer Perceptron (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    'XGBoost Classifier': XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, random_state=42, eval_metric='logloss'),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42)
}

results = []
oof_preds = {}
oof_probas = {}

for name, clf in models.items():
    start_t = time.time()
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', clf)
    ])
    
    y_pred = cross_val_predict(pipe, X, y_true, cv=cv, method='predict')
    y_proba = cross_val_predict(pipe, X, y_true, cv=cv, method='predict_proba')[:, 1]
    elapsed = round(time.time() - start_t, 2)
    
    oof_preds[name] = y_pred
    oof_probas[name] = y_proba
    
    acc = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred), 4)
    rec = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_proba), 4)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC-AUC': auc,
        'Time (s)': elapsed
    })

# Genuine Soft Voting Ensemble of top models
sve_start = time.time()
top_probas = (oof_probas['Random Forest'] + oof_probas['Gradient Boosting Classifier'] + oof_probas['LightGBM'] + oof_probas['Logistic Regression'] + oof_probas['CatBoost Classifier']) / 5.0
top_preds = (top_probas >= 0.5).astype(int)
sve_elapsed = round(time.time() - sve_start, 2)

oof_preds['Soft Voting Ensemble'] = top_preds
oof_probas['Soft Voting Ensemble'] = top_probas

results.append({
    'Model': 'Soft Voting Ensemble',
    'Accuracy': round(accuracy_score(y_true, top_preds), 4),
    'Precision': round(precision_score(y_true, top_preds), 4),
    'Recall': round(recall_score(y_true, top_preds), 4),
    'F1 Score': round(f1_score(y_true, top_preds), 4),
    'ROC-AUC': round(roc_auc_score(y_true, top_probas), 4),
    'Time (s)': sve_elapsed
})

res_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print("\n=====================================================================================")
print("GENUINE SCIKIT-LEARN 10-FOLD CV RESULTS (ALL 10 MODELS + SOFT VOTING)")
print("=====================================================================================")
print(res_df.to_string(index=False))

# Save results
with open('results_ml_summary.txt', 'w', encoding='utf-8') as f:
    f.write(res_df.to_string(index=False))

best_model_name = res_df.iloc[0]['Model']
best_preds = oof_preds[best_model_name]

cm = confusion_matrix(y_true, best_preds)
np.savetxt('results_confusion_matrix.txt', cm, fmt='%d')
print("\n=====================================================================================")
print(f"GENUINE CONFUSION MATRIX ({best_model_name}):")
print("=====================================================================================")
print(cm)

# -----------------------------------------------------------------------------------
# EXTENDED MCNEMAR STATISTICAL COMPARISONS ACROSS TOP MODELS
# -----------------------------------------------------------------------------------
def calculate_mcnemar(y_true, preds_a, preds_b):
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    
    b = np.sum(~correct_a & correct_b) # Misclassified by A, correct by B
    c = np.sum(correct_a & ~correct_b) # Correct by A, misclassified by B
    
    if (b + c) == 0:
        return b, c, 0.0, 1.0
        
    stat = ((abs(b - c) - 1)**2) / (b + c)
    p_val = chi2.sf(stat, df=1)
    return b, c, stat, p_val

# Add majority baseline predictions (all 0s)
oof_preds['Majority Baseline'] = np.zeros_like(y_true)

print("\n=====================================================================================")
print("PAIRWISE MCNEMAR'S TEST SIGNIFICANCE COMPARISONS")
print("=====================================================================================")
pairs_to_test = [
    ('Random Forest', 'Majority Baseline'),
    ('Soft Voting Ensemble', 'Majority Baseline'),
    ('Random Forest', 'Decision Tree'),
    ('Soft Voting Ensemble', 'Decision Tree'),
    ('Logistic Regression', 'Decision Tree'),
    ('Random Forest', 'Logistic Regression'),
    ('Soft Voting Ensemble', 'Logistic Regression'),
    ('Soft Voting Ensemble', 'Random Forest'),
    ('CatBoost Classifier', 'Logistic Regression'),
    ('Random Forest', 'CatBoost Classifier')
]

mcnemar_summary = []
for m1, m2 in pairs_to_test:
    b, c, stat, p = calculate_mcnemar(y_true, oof_preds[m1], oof_preds[m2])
    sig = "Statistically Significant (p < 0.05)" if p < 0.05 else "Not Significant (p >= 0.05)"
    out_str = f"Pair: {m1} vs {m2} -> b={b}, c={c}, McNemar Chi2: {stat:.4f}, p-value: {p:.6f} ({sig})"
    print(out_str)
    mcnemar_summary.append(out_str)

# Save results including McNemar summary
with open('results_ml_summary.txt', 'w', encoding='utf-8') as f:
    f.write(res_df.to_string(index=False))
    f.write("\n\n=====================================================================================\n")
    f.write("MCNEMAR SIGNIFICANCE COMPARISONS\n")
    f.write("=====================================================================================\n")
    f.write("\n".join(mcnemar_summary) + "\n")


# -----------------------------------------------------------------------------------
# DECISION THRESHOLD OPTIMIZATION (MATCHING TABLE 4b IN PAPER)
# -----------------------------------------------------------------------------------
print("\n=====================================================================================")
print("DECISION THRESHOLD OPTIMIZATION & SENSITIVITY TUNING (RANDOM FOREST)")
print("=====================================================================================")
rf_probas = oof_probas['Random Forest']

threshold_results = []
for th in [0.50, 0.45, 0.42, 0.40, 0.38, 0.35, 0.30]:
    th_preds = (rf_probas >= th).astype(int)
    th_cm = confusion_matrix(y_true, th_preds)
    tn, fp, fn, tp = th_cm.ravel()
    
    acc = accuracy_score(y_true, th_preds)
    prec = precision_score(y_true, th_preds)
    rec = recall_score(y_true, th_preds)
    spec = tn / (tn + fp)
    f1 = f1_score(y_true, th_preds)
    
    threshold_results.append({
        'Threshold': th,
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall (Sensitivity)': round(rec * 100, 2),
        'Specificity': round(spec * 100, 2),
        'F1 Score': round(f1 * 100, 2),
        'True Positives': tp,
        'False Negatives': fn
    })

th_df = pd.DataFrame(threshold_results)
print(th_df.to_string(index=False))
