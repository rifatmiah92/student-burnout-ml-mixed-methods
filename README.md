# Explainable Machine Learning for Student Burnout Classification and Risk Stratification

### A Mixed-Methods Study with Qualitative Triangulation

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: 100%](https://img.shields.io/badge/reproducibility-100%25-brightgreen.svg)]()
[![Mixed Methods: QUAN→QUAL](https://img.shields.io/badge/mixed--methods-QUAN%E2%86%92QUAL-purple.svg)]()
[![Live App](https://img.shields.io/badge/Live%20App-Burnout%20Radar-orange.svg)](https://burnoutwebapplication.netlify.app/)

**Live Web Application (Burnout Radar):** [https://burnoutwebapplication.netlify.app/](https://burnoutwebapplication.netlify.app/)  
**Web App Source Repository:** [https://github.com/rifatmiah92/Burnout-web-application](https://github.com/rifatmiah92/Burnout-web-application)

---

**Authors:** Rifat Miah<sup>1,*</sup> · Dr. A.S.M. Shihavuddin<sup>2</sup>

<sup>1</sup> Department of Computer Science and Engineering, Presidency University, Dhaka-1212, Bangladesh  
<sup>2</sup> Department of Electrical and Electronic Engineering, Green University of Bangladesh, Dhaka-1207, Bangladesh  
<sup>*</sup> Corresponding author: rifatmiah1992003@gmail.com

---

## Overview

This repository accompanies our mixed-methods research study investigating academic burnout among Bangladeshi university students. The work combines a large-scale supervised machine learning analysis with in-depth qualitative interviewing, aiming to understand not just *whether* burnout can be predicted, but *why* it occurs — from the students' own perspectives.

The full research pipeline is here: raw survey data, all preprocessing and feature engineering code, machine learning training and evaluation scripts, SHAP explainability analysis, anonymized qualitative interview transcripts, and the manuscript itself.

---

## What This Study Does

Most burnout studies either rely on simple questionnaire scoring or black-box ML models without any attempt at human explanation. We tried to do something more rigorous on both ends.

On the quantitative side, we collected primary survey data from 601 students across 11 Bangladeshi universities (8 private, 1 public, 2 National University colleges), trained and evaluated 10 classification algorithms under nested 10-fold cross-validation with strict data leakage controls, and optimized decision thresholds for clinical sensitivity. Rather than using raw questionnaire items directly, we engineered nine theoretically grounded composite features derived from Conservation of Resources (COR) and Job Demands–Resources (JD-R) frameworks — things like sleep deprivation index, screen-to-sleep ratio, and burnout vulnerability index.

On the qualitative side, we purposively selected 20 participants representing all burnout severity tiers, conducted semi-structured interviews, and ran reflexive thematic analysis (Braun & Clarke framework) with 25% back-translation audit and independent inter-rater verification (Cohen's κ = 0.82). The qualitative findings were used to contextualise and explain the computational patterns — not as an afterthought, but as a core part of the design.

---

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **65.89%** | **64.53%** | **43.53%** | **51.99%** | **0.7126** |
| Soft Voting Ensemble | 65.89% | 62.38% | 49.41% | 55.14% | 0.7069 |
| CatBoost | 65.06% | 60.87% | 49.41% | 54.55% | 0.6983 |
| Gradient Boosting | 64.39% | 59.62% | 49.80% | 54.27% | 0.6922 |
| Logistic Regression | 64.39% | 60.00% | 48.24% | 53.48% | 0.6819 |
| XGBoost | 62.40% | 56.44% | 49.80% | 52.92% | 0.6832 |
| Decision Tree | 61.40% | 56.28% | 40.39% | 47.03% | 0.6403 |
| *Majority Baseline* | *57.57%* | *—* | *0.00%* | *0.00%* | *0.500* |

Random Forest was the champion model. Compared to the majority-class baseline (57.57%), it achieved an 8.3 percentage-point improvement that is statistically significant (McNemar χ² = 13.96, *p* < .001). Nested cross-validation confirmed this result is not inflated (nested CV accuracy: 65.56%).

**Top SHAP predictors:** Academic performance index, CGPA bracket, screen-to-sleep ratio, and burnout vulnerability index — demographic variables (gender, age group) contributed minimally (mean |SHAP| < 0.007), which challenges assumptions that burnout is primarily demographics-driven.

At the calibrated decision threshold (θ = 0.38), sensitivity reached 71.76%, making the model practical for early-warning screening.

---

## Repository Contents

```
.
├── Manuscript_Student_Burnout.docx           # Full formatted manuscript (Word)
├── Quantitative_Survey_Data.xlsx             # Primary survey dataset (N = 601, fully anonymised)
├── Qualitative_Interview_Transcripts_Anonymized.pdf  # De-identified interview transcripts (N = 20)
├── Supplementary_S2_COREQ_Checklist.pdf      # COREQ reporting checklist for qualitative strand
├── Supplementary_S3_TRIPOD_Checklist.pdf     # TRIPOD reporting checklist for prediction model
├── ETHICS_STATEMENT_2.docx                   # Institutional ethics and participant consent statement
│
├── feature_engineering.py      # Canonical composite feature computation (9 indices)
├── train_ml.py                 # 10-fold CV benchmarking across 11 models + threshold tuning
├── run_shap.py                 # SHAP global importance + fold-by-fold stability verification
├── run_nested_cv.py            # Nested CV for unbiased hyperparameter tuning
├── pseudo_external_validation.py  # Cross-degree subgroup validation (bachelor vs postgrad)
├── export_model_to_js.py       # Exports trained RF + LR models to JSON for web app
├── generate_graphs.py          # High-resolution figures (Figures 2–6)
├── generate_flowchart.py       # Methodology flowchart (Figure 1)
│
├── results_ml_summary.txt      # Raw CV benchmark output
├── results_nested_cv.txt       # Nested CV results
├── results_shap_importance.txt # SHAP importance values + Spearman stability rho
├── results_pseudo_external.txt # Cross-subgroup validation output
│
├── Figure_1_Workflow.png       # End-to-end research methodology flowchart
├── Figure_2_Distribution.png   # Burnout severity distribution across sample
├── Figure_3_Gender.png         # Gender breakdown across burnout tiers
├── Figure_4_ML_Accuracies.png  # Model comparison bar chart
├── Figure_5_Confusion_Matrix.png  # Random Forest confusion matrix (θ = 0.38)
├── Figure_6_SHAP.png           # Global SHAP feature importance ranking
│
├── webapp/                     # Interactive client-side screening tool (zero server)
├── requirements.txt            # Pinned dependencies for exact reproducibility
└── LICENSE                     # MIT License
```

---

## Running the Code

Clone and set up the environment:

```bash
git clone https://github.com/rifatmiah92/student-burnout-ml-mixed-methods.git
cd student-burnout-ml-mixed-methods
pip install -r requirements.txt
```

Run the full ML benchmark:

```bash
python train_ml.py
```

Compute SHAP feature importances:

```bash
python run_shap.py
```

Pseudo-external cross-subgroup validation:

```bash
python pseudo_external_validation.py
```

Regenerate figures:

```bash
python generate_flowchart.py
python generate_graphs.py
```

All scripts read from `Quantitative_Survey_Data.xlsx` and write results to the corresponding `results_*.txt` files. Feature engineering is centralised in `feature_engineering.py` — both the Python pipeline and the JavaScript web app use the exact same mathematical formulas to ensure consistency.

---

## Ethics and Data Availability

This study was conducted in accordance with the Declaration of Helsinki. The research involved a non-invasive, minimal-risk cross-sectional survey and voluntary semi-structured interviews with enrolled university students. Formal ethics committee review was exempt under institutional guidelines for low-risk educational data research. Written informed electronic consent was obtained from all participants prior to data collection.

All survey data in this repository has been fully de-identified. Qualitative transcripts use standardised pseudonyms (P1–P20, Private University A–H, etc.). No personally identifiable information appears anywhere in the public files.

---

## Citation

If you use this dataset, code, or findings in your own work, please cite:

```bibtex
@article{miah2026burnout,
  title   = {Explainable Machine Learning for Student Burnout Classification 
             and Risk Stratification: A Mixed-Methods Study with 
             Qualitative Triangulation},
  author  = {Miah, Rifat and Shihavuddin, A.S.M.},
  year    = {2026},
  url     = {https://github.com/rifatmiah92/student-burnout-ml-mixed-methods}
}
```

---

## License

This repository is released under the MIT License. See `LICENSE` for details.
