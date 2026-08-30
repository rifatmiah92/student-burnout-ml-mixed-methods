# Explainable Machine Learning for Student Burnout Classification and Risk Stratification: A Mixed-Methods Study with Qualitative Triangulation

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: 100%](https://img.shields.io/badge/reproducibility-100%25-brightgreen.svg)]()
[![Methodology: QUAN%E2%86%92QUAL](https://img.shields.io/badge/mixed--methods-QUAN%E2%86%92QUAL-purple.svg)]()
[![Live Web Portal: Netlify](https://img.shields.io/badge/Live%20App-Burnout%20Radar-orange.svg)](https://burnoutwebapplication.netlify.app/)

> 🌐 **Live Web Application (Burnout Radar):** [https://burnoutwebapplication.netlify.app/](https://burnoutwebapplication.netlify.app/)  
> 💻 **Web Application Source Repository:** [https://github.com/rifatmiah92/Burnout-web-application](https://github.com/rifatmiah92/Burnout-web-application)

This repository contains the complete primary dataset, feature engineering pipeline, machine learning modeling codebase, Explainable AI (SHAP) evaluation scripts, anonymized qualitative interview transcripts, and manuscript sources for the study:

> **"Explainable Machine Learning for Student Burnout Classification and Risk Stratification: A Mixed-Methods Study with Qualitative Triangulation"**  
> **Authors:** Rifat Miah$^{1,*}$ and Dr. A.S.M. Shihavuddin$^{2}$  
> $^{1}$ Department of Computer Science and Engineering, Presidency University, Dhaka 1212, Bangladesh  
> $^{2}$ Department of Electrical and Electronic Engineering, Green University of Bangladesh, Dhaka 1207, Bangladesh  
> $^{*}$ Corresponding Author: `rifatmiah1992003@gmail.com`

---

## 📌 Abstract Overview

Academic burnout is a growing mental health concern in higher education, disproportionately affecting undergraduates in resource-constrained South Asian universities. This study employs an **explanatory sequential mixed-methods design (QUAN → QUAL)** on a primary cross-sectional dataset of $N = 601$ Bangladeshi undergraduates. 

1. **Quantitative Phase (QUAN):** Ten supervised machine learning classifiers and a Soft Voting Ensemble were evaluated under **10-fold stratified cross-validation** alongside nine domain-engineered composite features operationalizing Conservation of Resources (COR) and Job Demands-Resources (JD-R) theoretical constructs. Random Forest achieved top classification performance (Accuracy = 65.89%, ROC-AUC = 0.7126) — a statistically meaningful ~8.3 percentage-point improvement over the 57.57% majority baseline (McNemar $p < .001$).
2. **Explainable AI (SHAP):** SHAP global feature importances identified academic performance index, CGPA midpoint, and screen-to-sleep ratio as dominant burnout predictors, while demographic features contributed negligibly (mean $|SHAP| < 0.007$).
3. **Qualitative Phase (QUAL):** Semi-structured interviews ($N = 20$ participants purposively selected across survey burnout tiers) were analyzed via reflexive thematic analysis (Braun & Clarke framework, 25% back-translation audit, Cohen's $\kappa = 0.82$ inter-rater reliability) to explain, contextualize, and elaborate upon computational predictions.

---

## 📂 Repository Structure

```
.
├── Manuscript_Student_Burnout.md             # Complete manuscript source (Markdown)
├── Manuscript_Student_Burnout.docx           # Publication-ready Word document (Formatted)
├── Quantitative_Survey_Data.xlsx             # Primary survey dataset (N = 601)
├── Qualitative_Interview_Transcripts_Anonymized.pdf # De-identified qualitative transcripts (N = 20)
├── requirements.txt                          # Pinned dependency environment for 100% reproducibility
├── feature_engineering.py                    # Modular feature engineering pipeline (9 composite indices)
├── train_ml.py                               # 10-fold CV ML benchmarking & threshold optimization script
├── run_shap.py                               # SHAP global & local interpretability script
├── generate_flowchart.py                     # Flowchart generator (Figure 1)
├── generate_graphs.py                        # High-resolution figure generator (Figures 2-6)
├── pseudo_external_validation.py             # Subgroup pseudo-external validation script
├── convert_md_to_docx.py                     # Markdown-to-Docx converter script
├── Figure_1_Workflow.png                     # End-to-end methodology flowchart (600 DPI)
├── Figure_2_Distribution.png                 # Target variable severity distribution
├── Figure_3_Gender.png                       # Gender distribution across burnout levels
├── Figure_4_ML_Accuracies.png                # Model accuracy comparison benchmark
├── Figure_5_Confusion_Matrix.png             # Champion Random Forest confusion matrix
├── Figure_6_SHAP.png                         # SHAP global feature importance ranking
├── results_ml_summary.txt                    # Raw numerical cross-validation benchmark log
├── results_confusion_matrix.txt              # Champion model confusion matrix log
├── results_shap_importance.txt               # Raw SHAP numerical feature importances
└── results_pseudo_external.txt               # Cross-degree subgroup validation log
```

---

## 📊 Benchmark Machine Learning Results

Evaluated under 10-Fold Stratified Cross-Validation ($N = 601$, fixed `random_state=42`):

| Model / Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Time (s) |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.6589** | **0.6453** | **0.4353** | **0.5199** | **0.7126** | **10.99** |
| **Soft Voting Ensemble** | 0.6589 | 0.6238 | 0.4941 | 0.5514 | 0.7069 | 10.50 |
| CatBoost Classifier | 0.6506 | 0.6087 | 0.4941 | 0.5455 | 0.6983 | 11.46 |
| Logistic Regression | 0.6439 | 0.6000 | 0.4824 | 0.5348 | 0.6819 | 1.17 |
| Gradient Boosting | 0.6439 | 0.5962 | 0.4980 | 0.5427 | 0.6922 | 8.07 |
| Extra Trees Classifier | 0.6356 | 0.6139 | 0.3804 | 0.4697 | 0.6997 | 6.30 |
| Support Vector Machine (SVM) | 0.6356 | 0.6154 | 0.3765 | 0.4672 | 0.6708 | 3.38 |
| LightGBM | 0.6339 | 0.5785 | 0.5059 | 0.5397 | 0.6898 | 4.84 |
| XGBoost Classifier | 0.6240 | 0.5644 | 0.4980 | 0.5292 | 0.6832 | 10.11 |
| Decision Tree | 0.6140 | 0.5628 | 0.4039 | 0.4703 | 0.6403 | 1.06 |
| Multilayer Perceptron (MLP) | 0.6040 | 0.5359 | 0.4980 | 0.5163 | 0.6474 | 28.29 |
| **Majority Baseline** | **0.5757** | **—** | **0.0000** | **0.0000** | **0.5000** | **—** |

---

## ⚡ Quick Start & Reproducibility Guide

### 1. Environment Setup
Clone the repository and install the exact verified dependencies:
```bash
git clone https://github.com/rifatmiah92/student-burnout-ml-mixed-methods.git
cd student-burnout-ml-mixed-methods
pip install -r requirements.txt
```

### 2. Execute Machine Learning Training & Evaluation
Run 10-fold stratified cross-validation across all 10 models and decision threshold optimization:
```bash
python train_ml.py
```

### 3. Compute SHAP Feature Importance & Visualizations
Generate global and local SHAP attributions:
```bash
python run_shap.py
```

### 4. Regenerate High-Resolution Figures & Flowcharts
```bash
python generate_flowchart.py
python generate_graphs.py
```

---

## 📜 Citation & Ethical Statement

This study was conducted in accordance with the Declaration of Helsinki. Because the research involved a non-invasive, minimal-risk observational survey ($N=601$) and voluntary interviews ($N=20$) with enrolled university students (including a freshman transition bracket aged 17–18 years), formal institutional ethics committee review was exempt under institutional guidelines for minimal-risk educational data research. Informed electronic consent was explicitly obtained from all survey and qualitative participants prior to participation.

If you use this dataset or codebase in your research, please cite:

```bibtex
@article{miah2026burnout,
  title={Explainable Machine Learning for Student Burnout Classification and Risk Stratification: A Mixed-Methods Study with Qualitative Triangulation},
  author={Miah, Rifat and Shihavuddin, A.S.M.},
  journal={Educational Data Mining \& Mental Health Informatics},
  year={2026},
  publisher={GitHub Repository},
  url={https://github.com/rifatmiah92/student-burnout-ml-mixed-methods}
}
```

---

## 📄 License
This repository is open-sourced under the **MIT License**.
