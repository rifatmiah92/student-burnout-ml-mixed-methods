# 🎓 Student Burnout Prediction using Machine Learning

### A Research Study that Combines Machine Learning + Real Student Interviews

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: 100%](https://img.shields.io/badge/reproducibility-100%25-brightgreen.svg)](./)
[![Mixed Methods: QUAN→QUAL](https://img.shields.io/badge/mixed--methods-QUAN%E2%86%92QUAL-purple.svg)](./)
[![Live App](https://img.shields.io/badge/Live%20App-Burnout%20Radar-orange.svg)](https://burnoutwebapplication.netlify.app/)

**🌐 Live Web Application (Burnout Radar):** [https://burnoutwebapplication.netlify.app/](https://burnoutwebapplication.netlify.app/)  
**💻 Web App Source Code:** [https://github.com/rifatmiah92/Burnout-web-application](https://github.com/rifatmiah92/Burnout-web-application)

---

**Authors:** Rifat Miah<sup>1,*</sup> · Dr. A.S.M. Shihavuddin<sup>2</sup>

<sup>1</sup> Department of Computer Science and Engineering, Presidency University, Dhaka-1212, Bangladesh  
<sup>2</sup> Department of Electrical and Electronic Engineering, Green University of Bangladesh, Dhaka-1207, Bangladesh  
<sup>*</sup> Corresponding author: rifatmiah1992003@gmail.com

---

## 👋 What Is This Project?

This project is a **research study** about **academic burnout** — the feeling of being exhausted, unmotivated, and overwhelmed that many university students experience.

We wanted to answer two big questions:
1. **Can a computer (machine learning model) predict which students are at high risk of burnout?**
2. **Why does burnout happen — from the students' own words?**

To answer these questions, we:
- Surveyed **601 university students** across 11 Bangladeshi universities
- Trained **10 different machine learning (ML) models** to predict burnout
- Interviewed **20 students** in-depth to understand the human side of burnout

> **In simple terms:** We built an AI that can detect if a student is likely burned out, and we also listened to real students explain why burnout happens.

---

## 🗂️ What's Inside This Repository?

### 📄 Research Documents

| File | What it contains |
|------|-----------------|
| `Manuscript_Student_Burnout.pdf` | The full research paper (best place to start reading) |
| `Manuscript_Student_Burnout.docx` | Same paper in Word format (editable) |
| `Qualitative_Interview_Transcripts_Anonymized.pdf` | Real interview transcripts (student names hidden for privacy) |
| `Qualitative_Interview_Transcripts_Anonymized.docx` | Word version of interview transcripts |

### 📊 Data Files

| File | What it contains |
|------|-----------------|
| `Quantitative_Survey_Data.xlsx` | The survey data from 601 students (all personal info removed) |

### 📋 Ethics & Compliance Documents

| File | What it contains |
|------|-----------------|
| `ETHICS_STATEMENT_2.docx` | Ethics approval document for this study |
| `Ethics_Exemption_Letter_Template.docx` | Template for ethics exemption |
| `Supplementary_S1_COREQ_Checklist.pdf` | Checklist for qualitative (interview) research quality |
| `Supplementary_S1_COREQ_Checklist.docx` | Word version |
| `Supplementary_S2_TRIPOD_Checklist.pdf` | Checklist for prediction model quality |
| `Supplementary_S2_TRIPOD_Checklist.docx` | Word version |

### 🐍 Python Code Files

These are the scripts that run the machine learning analysis. You can run them yourself to reproduce our results.

| File | What it does |
|------|-------------|
| `feature_engineering.py` | **Start here.** Calculates 9 custom stress scores from raw survey answers (e.g., Sleep Deprivation Index, Academic Pressure Index). Used by all other scripts. |
| `train_ml.py` | **Main ML script.** Trains all 10 ML models + Soft Voting Ensemble and compares their performance |
| `run_shap.py` | Figures out **which factors matter most** for predicting burnout using SHAP explainability |
| `run_nested_cv.py` | Double-checks that our results are not inflated or "too good to be true" |
| `pseudo_external_validation.py` | Tests if the model works across different student groups (bachelor vs postgraduate) |
| `generate_graphs.py` | Creates the charts and figures used in the paper (Figures 2–6) |
| `generate_flowchart.py` | Creates the research workflow diagram (Figure 1) |

### 🤖 Machine Learning Models Used

All 10 models below are trained and compared inside `train_ml.py`:

| # | Model Name | Simple Explanation |
|---|-----------|-------------------|
| 1 | **Random Forest** 🏆 | Many decision trees voting together — our **best performing model** |
| 2 | **Gradient Boosting Classifier** | Trees that learn from each other's mistakes step by step |
| 3 | **LightGBM** | A very fast version of gradient boosting, good for large data |
| 4 | **CatBoost Classifier** | Another gradient boosting method, handles categories well |
| 5 | **XGBoost Classifier** | Powerful boosting model, very popular in competitions |
| 6 | **Extra Trees Classifier** | Like Random Forest but with more randomness, often faster |
| 7 | **Logistic Regression** | A classic, simple model — good baseline for comparison |
| 8 | **Decision Tree** | Makes decisions by splitting data like a flowchart |
| 9 | **Support Vector Machine (SVM)** | Finds the best boundary line to separate burnout vs no-burnout |
| 10 | **Multilayer Perceptron (MLP)** | A small neural network (like a tiny brain!) |
| + | **Soft Voting Ensemble** | Combines predictions from the top 5 models for better accuracy |

### 📈 Result Files (Auto-generated when you run the code)

| File | What it contains |
|------|-----------------|
| `results_ml_summary.txt` | Accuracy, F1, AUC scores for all 10 models + McNemar significance tests |
| `results_nested_cv.txt` | Nested cross-validation results (bias check) |
| `results_shap_importance.txt` | SHAP feature importance values + Spearman stability score |
| `results_pseudo_external.txt` | How well the model works on different student groups |
| `results_calibration.txt` | Brier Score + calibration curve (TRIPOD Item 22) |
| `results_confusion_matrix.txt` | Confusion matrix for the best model (Random Forest) |

### 🖼️ Figures (Charts & Diagrams)

| File | What it shows |
|------|--------------|
| `Figure_1_Workflow.png` | Step-by-step research process diagram |
| `Figure_2_Distribution.png` | How burnout levels are spread across all 601 students |
| `Figure_3_Gender.png` | Burnout rates by gender |
| `Figure_4_ML_Accuracies.png` | Bar chart comparing all 10 model accuracies |
| `Figure_5_Confusion_Matrix.png` | How the best model (Random Forest) classified students |
| `Figure_6_SHAP.png` | Which factors influence burnout the most |

### 🌐 Web Application

| Folder | What it contains |
|--------|-----------------|
| `webapp/` | Interactive burnout risk checker — runs entirely in the browser, no server needed |

---

## 🏆 How Well Did Our Models Perform?

Here are the results. **Higher is better** for Accuracy, F1, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| 🥇 **Random Forest** | **65.89%** | **64.53%** | **43.53%** | **51.99%** | **0.7126** |
| Soft Voting Ensemble | 65.89% | 62.38% | 49.41% | 55.14% | 0.7069 |
| CatBoost | 65.06% | 60.87% | 49.41% | 54.55% | 0.6983 |
| Gradient Boosting | 64.39% | 59.62% | 49.80% | 54.27% | 0.6922 |
| Logistic Regression | 64.39% | 60.00% | 48.24% | 53.48% | 0.6819 |
| XGBoost | 62.40% | 56.44% | 49.80% | 52.92% | 0.6832 |
| Decision Tree | 61.40% | 56.28% | 40.39% | 47.03% | 0.6403 |
| *Majority Baseline (just guessing)* | *57.57%* | *—* | *0.00%* | *0.00%* | *0.500* |

> **What does this mean?** The Random Forest model is **8.3 percentage points** more accurate than just guessing the most common answer. This is statistically significant (McNemar χ² = 13.96, p < 0.001). It is not just luck!

**🔍 Most important factors for predicting burnout (SHAP analysis):**
1. **Academic Performance Index (API)** — how well a student is doing academically
2. **CGPA bracket** — grade point average range
3. **Screen-to-Sleep Ratio (SSR)** — how much screen time relative to sleep hours
4. **Burnout Vulnerability Index (BVI)** — combined stress and academic pressure score

> **Interesting finding:** Gender and age barely mattered (mean |SHAP| < 0.007). Burnout is not about *who you are* — it is about your academic environment and daily habits.

At the optimized decision threshold (θ = 0.38), the model detects **71.76%** of burned-out students — making it practical for early-warning screening in university counselling.

---

## 🚀 How to Run This Code Yourself

### Step 1 — Install Python

Download Python **3.11** or **3.12** from [python.org](https://www.python.org/downloads/). Python 3.11 is strongly recommended.

### Step 2 — Download/Clone this Repository

```bash
git clone https://github.com/rifatmiah92/student-burnout-ml-mixed-methods.git
cd student-burnout-ml-mixed-methods
```

Or just download the ZIP file from GitHub and extract it.

### Step 3 — Install Required Libraries

```bash
pip install -r requirements.txt
```

This installs: scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, pandas, numpy, scipy, matplotlib, seaborn, and more.

### Step 4 — Run the Machine Learning Models

```bash
python train_ml.py
```

✅ Trains all 10 models + Soft Voting Ensemble → saves results to `results_ml_summary.txt` and `results_calibration.txt`

### Step 5 — (Optional) Run SHAP Explainability

```bash
python run_shap.py
```

✅ Computes which features matter most → saves to `results_shap_importance.txt`

### Step 6 — (Optional) Double-check with Nested Cross-Validation

```bash
python run_nested_cv.py
```

✅ Anti-inflation bias check → saves to `results_nested_cv.txt`

### Step 7 — (Optional) Test on Different Student Groups

```bash
python pseudo_external_validation.py
```

✅ Cross-group validation → saves to `results_pseudo_external.txt`

### Step 8 — (Optional) Regenerate the Figures

```bash
python generate_flowchart.py
python generate_graphs.py
```

✅ Re-creates all 6 high-resolution figures used in the paper

---

## 🔄 Reproducibility Verification

All experiments use fixed random seeds (`random_state=42`) and fully deterministic pipelines. The `requirements.txt` pins every library version. Running the scripts on any machine with the same environment will reproduce the reported numbers **exactly**.

| Claim in the Paper | How to verify |
|--------------------|--------------|
| RF Accuracy 65.89% | `python train_ml.py` → `results_ml_summary.txt` |
| Nested CV Accuracy 65.56% | `python run_nested_cv.py` → `results_nested_cv.txt` |
| Top SHAP feature API = 0.0388 | `python run_shap.py` → `results_shap_importance.txt` |
| Pseudo-external AUC 0.6597 | `python pseudo_external_validation.py` → `results_pseudo_external.txt` |
| Brier Score (calibration) | `python train_ml.py` → `results_calibration.txt` |

---

## 🛡️ Privacy & Ethics

This study was done responsibly:
- All 601 survey responses are **fully anonymized** — no names or personal details anywhere
- Interview participants use pseudonyms like P1, P2 … P20 and "Private University A, B, …"
- All students gave **voluntary informed consent** before participating
- The study follows the **Declaration of Helsinki** ethical standards
- Ethics was reviewed under institutional guidelines for low-risk educational research

See `ETHICS_STATEMENT_2.docx` for the full ethics statement.

---

## 📚 Citation

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

## 📜 License

This project is released under the **MIT License** — you are free to use, share, and build upon this work with proper credit. See the `LICENSE` file for full details.
