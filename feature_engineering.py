import pandas as pd
import numpy as np

def prepare_features(df):
    """
    Canonical Feature Engineering Module for Student Burnout Prediction.
    Ensures 100% code hygiene and identical feature definitions across
    all modeling, cross-validation, and SHAP interpretability scripts.
    """
    X = df.copy()
    
    # Target Binarization (High Burnout = 1, Low/Med = 0)
    if 'burnout_score' in X.columns:
        X['target'] = (X['burnout_score'] == 3).astype(int)
    
    # Clean Unweighted Domain Composite Features (Table 2 in Paper)
    X['psychological_strain_index'] = X['stress_score'] + X['depression_score']
    X['academic_pressure_index'] = X['academic_pressure_score'] + X['workload_score']
    X['burnout_vulnerability_index'] = (X['psychological_strain_index'] * X['academic_pressure_index']) / (X['motivation_score'] + X['sleep_quality_score'] + 0.1)
    # Guard: ensure non-negative result even if sleep hours or quality scores fluctuate
    X['sleep_deprivation_index'] = np.maximum(0.0, (8.0 - X['sleep_hours_numeric']) * (4.0 - X['sleep_quality_score']))
    X['screen_to_sleep_ratio'] = X['social_media_hours'] / (X['sleep_hours_numeric'] + 0.1)
    X['study_to_rest_ratio'] = (X['study_hours_numeric'] + X['social_media_hours']) / (X['sleep_hours_numeric'] + X['physical_activity_hours'] + 0.1)
    X['academic_performance_index'] = (X['cgpa_midpoint'] / 4.0) * (X['attendance_pct'] / 100.0)
    X['motivation_deficit_score'] = (4.0 - X['motivation_score']) * X['stress_score']
    X['wellbeing_buffer'] = (X['physical_activity_hours'] + X['sleep_quality_score']) - X['stress_score']
    
    return X


if __name__ == '__main__':
    df = pd.read_excel('Quantitative_Survey_Data.xlsx')
    df_feat = prepare_features(df)
    print(f"Canonical Feature Engineering Module verified! Total features created: {df_feat.shape[1]}")
