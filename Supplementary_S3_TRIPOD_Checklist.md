# Supplementary Material S3
## TRIPOD Checklist — Transparent Reporting of a Multivariable Prediction Model
### for Individual Prognosis Or Diagnosis (TRIPOD) Statement
### Collins et al. (2015) — 22-Item Checklist for Prediction Model Development Studies

**Paper Title:** Explainable Machine Learning for Student Burnout Classification and Risk Stratification: A Mixed-Methods Study with Qualitative Triangulation

**Authors:** Rifat Miah (242370038@student.presidency.edu.bd), Dr. A.S.M. Shihavuddin (shihav@eee.green.edu.bd)

**Corresponding Author:** Rifat Miah (Email: 242370038@student.presidency.edu.bd), Department of Computer Science and Engineering, Presidency University, Dhaka 1212, Bangladesh (Institutional Contact: info@pu.edu.bd)

**Model Type:** Classification (Prediction Model Development — 10-Fold Stratified Cross-Validation)

**Outcome:** Binary classification of High Academic Burnout (burnout_score = 3) vs. Low/Medium Burnout (burnout_score = 1 or 2) in Bangladeshi university students.

---

## TRIPOD Checklist

| Section | Item No. | Checklist Item | Manuscript Location | Reported? |
|---------|----------|----------------|---------------------|-----------|
| **Title and Abstract** | 1 | Identify the study as developing and/or validating a multivariable prediction model, the target population, and the outcome to be predicted. | Title, Abstract (Methods paragraph) | ✅ Yes |
| **Introduction** | 2a | Explain the medical context (including whether diagnostic or prognostic) and rationale for developing or validating the multivariable prediction model, including references to existing models. | Section 1.1, Section 2.4, Table 1 | ✅ Yes |
| | 2b | Specify the objectives, including whether the study describes the development or validation of the model or both. | Section 1.7, Section 1.4 (H1–H4) | ✅ Yes |
| **Methods — Source of Data** | 3 | Describe the study design or source of data (e.g., randomised trial, cohort, or registry data), separately for the development and validation data sets, if applicable. | Section 3.1, Section 3.2 | ✅ Yes |
| | 4 | Describe the eligibility criteria for participants. | Section 3.2, Section 3.4 (Ethics; 18+ enrolled students) | ✅ Yes |
| | 5 | Describe the outcome(s) to be predicted. | Section 3.3, Section 3.3.1, Table 2 (burnout_score variable) | ✅ Yes |
| | 6 | Report the number of participants with and without the outcome. | Section 3.5 (High: n = 255, 42.43%; Low/Med: n = 346, 57.57%) | ✅ Yes |
| | 7 | Describe the predictors used in the model, including how and when they were measured. | Section 3.3 (Table 2), Section 3.5 (Table 3, composite indices) | ✅ Yes |
| | 8 | Describe the sample size, including any sample size calculations. | Section 3.2 (N = 601); Appendix / power note in Section 3.2 (G*Power calculation referenced, minimum n = 452 per Cohen, 1988 [118]) | ✅ Yes |
| | 9 | Describe how missing data were handled. | Section 3.5 ("no missing values for any of the 18 variables") | ✅ Yes |
| **Methods — Model Development** | 10a | Describe how predictors were handled in the analyses. | Section 3.5 (StandardScaler, OneHotEncoder, leak-free pipeline), Section 3.6 | ✅ Yes |
| | 10b | Specify the type of model, all model-building procedures (including any predictor selection), and method for internal validation. | Section 3.6 (10 classifiers + Soft Voting Ensemble described; GridSearchCV nested CV for hyperparameter selection) | ✅ Yes |
| | 10c | For validation, describe the method of performance assessment. | Section 3.6, Section 6 (10-fold stratified CV, McNemar test, Bonferroni correction; Table 4–Table 6) | ✅ Yes |
| **Methods — Statistics** | 11 | Describe all measures used to assess model performance and, if relevant, to compare multiple models. | Section 3.6, Section 6 (Accuracy, Balanced Accuracy, ROC-AUC, F1, Precision, Recall, MCC, Brier Score; McNemar pairwise tests reported in Table 5) | ✅ Yes |
| **Results — Participants** | 12 | Report the number of participants and outcome events in the overall data set and, if applicable, in the development and validation data sets. | Section 4.1, Section 6.1 (Table 4: n = 601; High Burnout n = 255; Low/Med n = 346) | ✅ Yes |
| | 13 | Describe the characteristics of the participants (basic demographics, clinical characteristics) and the distribution of predictors. | Section 4 (full EDA: Sections 4.1–4.4), Table 2, Table 3 | ✅ Yes |
| **Results — Model Development** | 14a | Specify the number of participants and outcome events in each analysis. | Section 6.1, Table 4 | ✅ Yes |
| | 14b | If done, report the results from any model selection procedure. | Section 3.6 (all 10 models benchmarked; champion model selected by Accuracy+AUC; nested CV hyperparameter tuning described) | ✅ Yes |
| | 15 | Present the full prediction model to allow predictions for individuals (i.e., all regression coefficients, and model intercept or baseline survival at a given time point). | Section 7 (SHAP global feature importance Table 7a, SHAP local attributions Figure 6); the full trained Random Forest model is available at the GitHub repository. *Note: For tree-ensemble models, the equivalent of regression coefficients is the full SHAP value decomposition, which is reported in Table 7a and Figure 6.* | ✅ Yes |
| | 16 | Report performance measures of the prediction model. | Section 6.1 (Table 4: 10-fold CV metrics for all models); Section 6.2 (confusion matrix analysis, threshold calibration) | ✅ Yes |
| **Discussion** | 17 | Discuss the limitations of the study (e.g., nonrepresentative sample, few events per variable, missing data). | Section 10 (full limitations section: single-item criterion, cross-sectional design, convenience sampling, geographic concentration, pseudo-external validation only, absence of pre-registration) | ✅ Yes |
| **Other Information** | 18 | Provide information about funding sources and the role of the funders for the present study. | *To be added to final manuscript:* This research received no external funding. The study was conducted as part of the academic research activities of the Department of Computer Science and Engineering, Presidency University. | ⚠️ To Add |
| | 19 | Provide supplementary information (e.g., study protocol, statistical analysis plan, or patient consent forms). | Supplementary Material S1 (Interview Guide), S2 (COREQ Checklist), S3 (TRIPOD Checklist) — GitHub repository link provided in Section 3.4. | ✅ Yes |

> **⚠️ Action Required for Item 18:** Add the following sentence to the manuscript (Acknowledgements section or footnote): *"This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors. The study was conducted as part of independent academic research activities."*

---

## Explanatory Notes on Key Items

### Item 10b — Model Building Procedure
Ten classification models were evaluated: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, Support Vector Machine, and Multi-layer Perceptron, plus a Soft Voting Ensemble. Hyperparameter tuning was performed via nested 10-fold outer / 5-fold inner cross-validation using `GridSearchCV` on the top models. All pipelines were implemented in scikit-learn with `ColumnTransformer` preprocessing to prevent data leakage.

### Item 15 — Full Prediction Model
For tree-ensemble models, SHAP (SHapley Additive exPlanations) values represent the functionally equivalent construct to regression coefficients, providing both global feature importance (Table 7a) and individual-level attributions (Figure 6). The full trained Random Forest model (serialised in `.pkl` format), all pipeline components, and SHAP explainer objects are publicly available at: https://github.com/rifatmiah92/student-burnout-ml-mixed-methods

### Item 16 — Performance Measures
Primary metrics: Accuracy, Balanced Accuracy, ROC-AUC, Macro-F1, Precision, Recall, Matthews Correlation Coefficient (MCC), Brier Score. All reported as mean ± SD across 10 folds. Pairwise McNemar tests with Bonferroni correction performed for all model comparisons.

---

## TRIPOD Compliance Summary

| Section | Items | Compliant | Partial/Action Required |
|---------|-------|-----------|------------------------|
| Title & Abstract | 1 | 1/1 | — |
| Introduction | 2 | 2/2 | — |
| Methods | 3–11 | 9/9 | — |
| Results | 12–16 | 5/5 | — |
| Discussion | 17 | 1/1 | — |
| Other | 18–19 | 1/2 | Item 18 (Funding statement to be added) |
| **Total** | **22** | **21/22** | **1 item to add (funding statement)** |

**Overall TRIPOD Compliance: 21/22 items (95%) — 1 minor addition required**

---

*Reference: Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): The TRIPOD statement. BMJ, 350, g7594. https://doi.org/10.1136/bmj.g7594*

*Moons, K. G., Altman, D. G., Reitsma, J. B., Ioannidis, J. P., Macaskill, P., Steyerberg, E. W., ... & Collins, G. S. (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): Explanation and elaboration. Annals of Internal Medicine, 162(1), W1–W73. https://doi.org/10.7326/M14-0698*
