# Explainable Machine Learning for Student Burnout Classification and Risk Stratification: A Mixed-Methods Study with Qualitative Triangulation

Rifat Miah<sup>1,*</sup> and Dr. A.S.M. Shihavuddin<sup>2</sup>

<sup>1</sup> Department of Computer Science and Engineering, Presidency University, Dhaka 1212, Bangladesh

<sup>2</sup> Department of Electrical and Electronic Engineering, Green University of Bangladesh, Dhaka 1207, Bangladesh

*Corresponding Author: Rifat Miah (Email: rifatmiah1992003@gmail.com)

Co-Author: Dr. A.S.M. Shihavuddin (Email: shihav@eee.green.edu.bd)

*Institutional Contact:* Presidency University, Dhaka 1212, Bangladesh (Email: info@pu.edu.bd)

*Affiliation Note*: Research activities, survey data collection, qualitative interviews, and predictive modeling were conducted at the Department of Computer Science and Engineering, Presidency University, Dhaka, Bangladesh.

## Abstract

**Background:** Academic burnout is a growing mental health challenge in South Asian higher education, where structural counselling infrastructure is scarce. Existing screening frameworks rely on isolated psychometric surveys without interpretable predictive modelling or qualitative contextualisation. *Important Note on Construct Validity:* The primary burnout outcome in this study is operationalised using a single-item global self-rating (not a validated multi-item instrument such as MBI-SS or CBI), which represents a fundamental methodological boundary that constrains clinical generalisability and limits direct performance comparisons with instrument-validated studies.

**Methods:** An explanatory sequential mixed-methods design (QUAN → QUAL) was implemented on a cross-sectional cohort of N = 601 university students across 11 institutions in Bangladesh. Ten supervised classification algorithms and a Soft Voting Ensemble were evaluated via 10-fold stratified cross-validation. Nine theory-grounded composite indices operationalising Conservation of Resources (COR) and Job Demands-Resources (JD-R) frameworks were included. A purposive nested sub-sample of N = 20 students across burnout severity tiers participated in semi-structured qualitative interviews analysed by reflexive thematic analysis.

**Results:** Random Forest achieved the highest classification performance (Accuracy = 65.89% ± 5.83%, ROC-AUC = 0.7126 ± 0.0775), representing a statistically significant 8.32 percentage-point gain over the 57.57% majority baseline (McNemar p < .001, Bonferroni-adjusted p = .001). Decision threshold calibration (th = 0.38) raised sensitivity to 71.76% for high-coverage voluntary screening. Global SHAP analysis ranked academic performance index, CGPA midpoint, and screen-to-sleep ratio as the primary risk predictors; sociodemographic variables contributed negligibly (mean |SHAP| < 0.007). Reflexive thematic analysis of 20 interview transcripts revealed four themes: academic dread and career despair, institutional identity and low-pressure burnout, digital fatigue as maladaptive coping, and circadian collapse.

**Conclusion:** This exploratory study demonstrates that integrating supervised machine learning, SHAP explainability, and reflexive qualitative analysis can identify theoretically coherent and contextually grounded burnout risk patterns in a South Asian university cohort. The findings are consistent with a hypothesised compounding pathway — career anxiety → psychological distress → digital escapism → biological sleep exhaustion — requiring longitudinal confirmation. These results constitute a transparent, non-binding exploratory baseline to inform campus mental health and mentoring systems, not a clinical diagnostic framework. An open-access interactive web screening portal is publicly accessible at: `https://burnoutwebapplication.netlify.app/`. **Data Availability:** Anonymised survey data and qualitative transcripts are available as Supplementary Materials.

Keywords: Academic Burnout, Explainable Artificial Intelligence (XAI), SHAP, Supervised Machine Learning, Explanatory Sequential Mixed-Methods, Student Mental Health, Educational Data Mining.

## 1. Introduction

### 1.1 Background of the Study

Academic burnout is a serious and increasing problem in modern higher education among university students [1]. Burnout is defined as chronic emotional exhaustion, depersonalisation or cynicism toward one’s coursework, and a diminishing sense of personal accomplishment [2], and goes beyond the typical academic fatigue. It shows up as systemic student disengagement with long-term consequences for academic persistence, degree completion and post-graduate health. Stress levels have been rising on university campuses across the world in recent years, due to heavier course loads, competitive assessment schemes, uncertainties regarding employment, and the ever-present digital distractions [3, 4]. 
The magnitude of this problem is documented in the international literature. The World Health Organization [5] recognised burnout as an occupational syndrome in ICD-11, recognising that unaddressed chronic stress leads to serious health consequences. While the conceptualisation of the framework was initially done in work environments [6], the framework is readily transferable to higher education, where university students face cognitive workloads, deadlines, and performance evaluations that are similar to professional demands [7]. Cross-national studies estimate that 30 to 50% of undergraduates suffer from clinically relevant burnout [8–10]. Rates of burnout are especially high in heavy-workload disciplines and developing economies. This challenge is further compounded by structural constraints in South Asia. Over the past two decades, university enrolment in Bangladesh has grown rapidly but institutional mental health services, student counselling centers and faculty mentoring programs have lagged [11, 12]. Recent empirical assessments by Hossain et al. [13] indicated that approximately 47% of surveyed Bangladeshi university students experience moderate-to-severe psychological distress, anxiety, and depressive symptoms, which are strongly correlated with academic disengagement and emotional exhaustion.
The need is clear, but burnout is difficult to identify before it leads to course failure or institutional dropout. Traditionally, universities have relied on retrospective, end-of-term evaluations—tools that document burnout only after emotional exhaustion and academic decline have already set in [14]. Furthermore, burnout is due to interdependent lifestyle habits, psychological vulnerabilities and institutional pressures which cannot be captured adequately by univariate cutoffs [15, 16]. This complexity requires predictive methods that can identify multivariate patterns across behavioural and academic indicators. 
Supervised machine learning (ML) offers an analytical framework for identifying risk signatures in observational student data [17, 18]. Predictive algorithms have shown promise in educational data mining and precision psychiatry predicting dropout risk, academic struggle and depressive symptoms [17–19]. Existing ML studies on student mental health suffer from two major shortcomings. First, most models are trained and tested on single institution datasets with only internal cross-validation, and thus generalisability across distinct student cohorts is unverified [20]. Second, the opacity of “black-box” models, without providing explanations, makes it difficult to understand why certain students are identified as high risk, which hinders their practical use by university counsellors and administrators, who need actionable explanations [21, 22]. 
Explainable artificial intelligence (XAI) frameworks, specifically SHapley Additive exPlanations (SHAP) based on cooperative game theory [23], overcome this interpretability barrier by quantifying the contribution of each feature to the overall model behaviour [21]. SHAP can also be used for transparent algorithmic auditing [22, 24, 25], quantifying the influence of academic metrics, sleep schedules and lifestyle factors on risk classifications while safeguarding algorithmic fairness and equity in educational applications [25]. 
Yet, the experience of academic stress in students' daily lives cannot be fully explained by statistical modelling alone. The numeric survey responses cannot provide the important context offered by narrative descriptions of employment challenges, sleep deprivation, smartphone use and institutional stigma [26]. Mixed-methods research designs that integrate quantitative predictive models with qualitative inquiry can yield triangulated insights with greater practical relevance than either method in isolation [27, 28]. However, there is a dearth of literature on student burnout that combines supervised machine learning with qualitative thematic analysis.

### 1.2 Statement of the Problem

Although the number of studies that investigated correlates of academic burnout is large [1, 26], the existing literature suffers from four major limitations. First, most of the machine learning applications are fully dependent on single source datasets without assessing the robustness on sub-populations [29]. Second, few educational models incorporate game-theoretic XAI techniques to go beyond simple feature counts to interpretable attributions [21, 22]. Third, there is a paucity of psychometric predictive research in developing countries where socio-economic realities, academic structures and counselling resources are very different from Western contexts [11, 12]. Fourth and most importantly, few studies integrate machine learning classification, SHAP interpretability, and qualitative thematic analysis in an explanatory sequential mixed-methods design (QUAN → QUAL) [22, 30, 31]. 
These gaps are particularly relevant in Bangladesh. Predictive models without qualitative validation risk misinterpreting local behavioural patterns and misallocating limited campus counselling resources. 
To address these obstacles, this study addresses the following overarching question: How can an explanatory sequential mixed-methods design—integrating supervised machine learning, SHAP feature interpretability, and qualitative interview analysis—develop an accurate, interpretable, and contextually grounded predictive framework for university student burnout?

### 1.3 Research Questions

More specifically, the investigation deals with the following four research questions in its quantitative, qualitative and integrative sections: 
RQ1. Which supervised machine learning algorithm provides the best predictive performance for binary student burnout classification (High vs. Low/Medium) based on 10-fold stratified cross-validation on primary survey data? 
RQ2. What are the most important predictors of burnout in terms of behavioural, psychological and academic features from the global SHAP feature importance analysis? 
RQ3. What are the main themes of university students’ lived experiences of academic burnout in semi-structured qualitative interviews and how do these narratives correspond with quantitative feature rankings? 
RQ4. What are the meta-inferences from the triangulation of algorithmic predictions, XAI feature attributions and qualitative student narratives, and what are the practical implications for campus mental health support?

### 1.4 Research Hypotheses

Based on the COR (Conservation of Resources) theory [32], the Job Demands-Resources (JD-R) model [33] and the previous empirical literature [1, 4], we specify the following four theory-derived, exploratory research expectations, formulated prior to data analysis. *Transparency Note:* These expectations are not formally pre-registered in a public registry (e.g., OSF, AsPredicted) and are therefore designated as *exploratory, theory-driven* hypotheses rather than confirmatory pre-registered predictions. Results should be interpreted accordingly and considered hypothesis-generating for future pre-registered replication studies:
H1 (Exploratory). Ensemble-based classifiers (e.g. Random Forest, XGBoost [34], LightGBM [35], CatBoost [36]) will perform better than single learner models (e.g. Logistic Regression, Decision Tree) on F1-scores and ROC-AUC metrics. 
H2 (Exploratory). The highest SHAP feature importance predictors would be academic performance indices and psychological demands (i.e., the academic performance index, the CGPA midpoint, the screen-to-sleep ratio, and the burnout vulnerability index). 
H3 (Exploratory). According to COR theory's resource-preservation principles, resource-related indicators (sleep quality, physical activity, wellbeing buffer) will be negatively associated with severity of burnout. 
H4 (Exploratory). Burnout is affected by several factors such as academic workload, digital fatigue, lack of sleep, and work demand. Qualitative themes from the student interviews will be leveraged to help identify major contributors to burnout and support the quantitative feature importance hierarchy.

### 1.5 Conceptual Framework

In this study, an explanatory sequential mixed-methods design (QUAN → QUAL) is utilised with two primary analytical phases: (1) Quantitative supervised machine learning with SHAP explainability, and (2) Qualitative reflexive thematic analysis. The conceptual framework describes burnout as an emergent phenomenon arising from five clusters of interacting factors:

• Academic Demands: Study hours, academic pressure and workload score.

• Psychological Vulnerability: Self-report of depression and stress.

• Lifestyle & Behavioural Habits: Sleep duration, sleep quality, physical activity, and social media consumption.

• Sociodemographic Characteristics: Age, gender, academic year, and level of degree.

• Structural Resources & Performance: Biological sleep recovery, Physical exercise, Motivation score, CGPA, attendance.

In the first quantitative phase (QUAN), these variables are evaluated across 10 supervised classifiers and a Soft Voting Ensemble through 10-fold stratified cross-validation on N = 601 survey records, followed by SHAP feature attribution. In the subsequent qualitative part (QUAL) a purposive sub-sample of N = 20 students from low, medium and high burnout tiers participates in semi-structured interviews. Triangulating both strands yields integrated meta-inferences about the structural mechanisms of student burnout.

### 1.6 Theoretical Framework

The choice of features and empirical interpretations are organised around three complementary theoretical frameworks.

#### 1.6.1 Theory of Conservation of Resources (COR)

As Hobfoll’s COR theory states [32, 37], psychological distress results from loss of resources, threat of loss, or failure to gain after significant resource investment. In college the key resources are restorative sleep, physical stamina, academic motivation and peer support, the key demands are heavy workloads, high-stakes grading and financial obligations. COR theory posits that a downward spiral in burnout is caused by the ongoing depletion of resources without replenishment [32, 37], which influences our interpretation of recovery measures like sleep quality and physical activity.

#### 1.6.2 Job Demands-Resources (JD-R) Model

The JD-R framework is a theoretical model adapted from the field of organisational psychology to educational contexts [33, 38]. This framework differentiates between demands (requirements that require sustained mental or emotional effort, such as academic pressure and screen time) and resources (protective factors that support engagement, such as sleep quality and intrinsic motivation). The model emphasises interaction effects: resources, when available, can buffer the deleterious impact of high demands [38, 39]. We use SHAP dependence plots and non-linear tree-based models in our pipeline to evaluate these interaction dynamics.

#### 1.6.3 Self-Determination Theory (SDT)

SDT [40, 41] of Deci and Ryan suggests that intrinsic motivation is determined by the satisfaction of three basic psychological needs: autonomy, competence and relatedness. When these needs are thwarted – when students feel trapped by inflexible curricula, doubt themselves due to poor grades, or lack supportive institutional networks – amotivation and exhaustion often follow [40]. SDT provides a lens into qualitative stories of despair about career and institutional disengagement. Theoretical Synthesis: These frameworks concur on the common premise that student burnout does not result from a single factor, but rather is the product of an imbalance of demands and resources (JD-R), cumulative depletion of restorative resources (COR), and unfulfilled psychological needs (SDT). The multivariate empirical patterns across these domains are explained by computational modelling, their relative influence is uncovered by XAI, and their contextual grounding is provided by qualitative narratives.

### 1.7 Aims of the Research

The study has four principal objectives: 1. Train and evaluate ten supervised classification algorithms and a Soft Voting Ensemble for binary burnout prediction on a primary survey dataset (N=601) under 10-fold stratified cross-validation. 2. Explain the decision mechanics and global feature importance hierarchy of the best performing model using SHAP (SHapley Additive exPlanations). 3. Conduct reflexive thematic analysis of semi-structured interviews with 20 university students to describe lived experiences of academic exhaustion. 4. Use a structured convergence protocol to triangulate quantitative and qualitative findings to construct actionable meta-inferences for campus mental health systems.

### 1.8 Importance of the Study

This work contributes to educational data mining and student mental health research in three major ways: Methodologically, it shows an end-to-end integration of supervised learning, XAI interpretability and qualitative thematic analysis in an explanatory sequential mixed-methods framework, overcoming the interpretability and contextual gaps of traditional ML studies [20, 42]. In principle, it tests propositions from COR, JD-R and SDT models against both statistical attributions and lived student accounts. Practically, the identification of primary risk factors and the calibration of screening thresholds provide evidence-based insight for university administrators involved in the design of early-warning dashboards and the allocation of student support resources in South Asian higher education [29].

### 1.9 Scope and Limitations

The quantitative survey comprises N = 601 undergraduate and postgraduate students of different Bangladeshi universities, collected through Google Forms. The qualitative strand includes N = 20 students sampled purposefully across levels of burnout, institution type and year of study. Although the findings are context-specific to higher education in Bangladesh, the integrated methodological framework is designed to be transferable to wider international higher education contexts.

### 1.10 Definitions of Key Terms

• Academic Burnout: The feeling of emotional exhaustion, cynicism to academic commitments and low efficacy from chronic educational demands [7]. • Explainable AI (XAI): Analytical techniques such as SHAP, feature attribution techniques explaining the predictive logic of complex machine learning algorithms [23]. • Explanatory Sequential Mixed-Methods: A research design where quantitative data collection and analysis (QUAN) precedes and informs qualitative investigation (QUAL) to explain empirical findings [26]. • SHAP (SHapley Additive exPlanations): A game theoretic approach that calculates the marginal contribution of each feature to individual and global model predictions [21]. • Triangulation: Systematic integration of quantitative statistical patterns and qualitative narratives to assess convergence, complementarity and divergence [26, 28].

### 1.11 Paper structure

The remainder of this paper is organised as follows. Section 2 discusses related work on student burnout, machine learning in education and explainability methods. Section 3 covers the materials and methods, including data collection, pre-processing, feature engineering, model training, XAI procedures, qualitative coding, and ethical safeguards. Section 4 gives the exploratory data analysis. Section 5 deals with the inferential statistics tests and multicollinearity. Section 6 presents cross-validated ML benchmarks, error analysis of the confusion matrix and calibration of decision thresholds. The SHAP explainability results are shown in Section 7. Section 8 presents qualitative thematic results. Mixed-methods triangulation and discussion are provided in Section 9. Section 10 concludes with policy recommendations and limitations of the study.

## 2. Literature Review

This section reviews the empirical literature on student burnout, behavioural and psychological risk factors, machine learning applications in educational data mining, explainable AI, and mixed methods research designs. The review highlights significant contributions and methodological shortcomings of previous studies and thus constitutes the empirical basis of the present study .

### 2.1 Concept and Operationalisation of Academic Burnout

Burnout is an occupational health construct first described in the clinical setting by Freudenberger [43], and later formally operationalised by Maslach and Jackson [44] as a three-dimensional syndrome of emotional exhaustion, depersonalisation (cynicism) and reduced personal accomplishment. The Maslach Burnout Inventory (MBI) remains the most commonly used measure in the work and organisational research literature [6, 28]. Given the evaluative pressure, competitive deadlines and cognitive demands that university students are exposed to, Schaufeli et al. [7] adapted the framework to be applicable to the higher education context through the Maslach Burnout Inventory-Student Survey (MBI-SS). The MBI-SS redefines the three critical dimensions of an academic environment: exhaustion (overwhelmed by the demands of study), cynicism (disaffected and disillusioned regarding the coursework), and reduced efficacy (diminishing confidence in one’s ability as a student) [7, 14]. This three-factor structure has been replicated in later psychometric studies using European [3, 7], North American [6, 45] and Asian [46] samples. Recent scholarship, however, conceptualises burnout more as a continuous behavioural and physiological process embedded in students’ daily routines such as sleep quality, physical activity, digital consumption, and social interactions [1, 47–49]. Behavioural features can be early objective markers for risk rather than burnout as an end state psychological score. Burnout has also been identified by researchers as a progressive continuum [33, 50] that supports the use of multi-tiered risk stratification (Low, Medium, High) for clinical and preventive screening [50, 51].

### 2.2 Incidence and Consequences in Higher Education

Burnout is a common problem for university students according to worldwide epidemiological surveys. In a systematic review of 15 countries, Frajerman et al. [8] estimated the prevalence of significant burnout symptoms in university students to be between 28% and 55%, higher in competitive STEM and medical studies [52, 53]. Meta-analytic data from Erschens et al. [9] shows that around 44% of medical students worldwide suffer from emotional exhaustion. In the post-pandemic period, many reviews reported high rates of burnout worldwide, for example 61% in Saudi Arabia [10], Turkey [54], India [19], Brazil [55–57] and Bangladesh [58]. There are particular structural barriers facing university students in South Asia. Hossain et al. [13] reported that over 47% of surveyed Bangladeshi undergraduates experienced moderate-to-severe psychological distress and depressive affect during higher education coursework. Private institutions had high rates, where the demands of coursework are compounded by academic fees and employment pressures [59]. Systemic vulnerability was identified in multi-institutional reviews by Ali et al. [60] and Hossen et al. [30]. It was found that Bangladeshi students experienced pervasive anxiety and depressive affect [14, 61], which was caused by a lack of adequate mental health infrastructure on campus. Untreated burnout of students in higher education has well-documented effects. Longitudinal studies have shown that chronic burnout may predict academic underachievement [32], increased dropout intentions [62, 63], substance misuse [53, 55], depressive episodes and suicidal ideation [56, 64]. In developing countries, where educational investment represents a major financial commitment for families under rising economic pressures [11], academic attrition has profound socio-economic consequences for students and the community.


### 2.3 Predictors and Correlates of Student Burnout

The literature to date indicates contributing factors in four broad areas of analysis:

#### 2.3.1 Academic Workload and Demand-Side Predictors

“Academic strain is consistently associated with frequent examinations, rigid grading curves, and volume of course work” [1]. Salmela-Aro and Upadyaya [33] found that increased study demands predicted burnout trajectories in a longitudinal study, even when controlling for baseline psychological health. Subjective evaluation of workload manageability was a stronger predictor than the raw hours invested [65], indicating that cognitive appraisal acts as a mediator in the relationship between demands and exhaustion [4].

#### 2.3.2 Psychosocial factors

Academic burnout is often associated with stress, anxiety and depression [47, 48]. Meta-analytic results of Bianchi et al. [64] demonstrated considerable overlap of burnout with depressive symptoms (pooled r = .52). In addition, systematic vulnerability appears in individuals with high levels of trait anxiety [49], neuroticism [51, 66] and academic helplessness [52]. Financial problems have also been cited as a major stressor. Richardson et al. [67] found that students experiencing financial strain were 1.8 times more likely to report severe burnout and Walsemann et al. [68] reported accumulated education debt as a direct contributor to psychological strain.

#### 2.3.3 Lifestyle and behavioural factors

Of the behavioural determinants of student wellbeing, sleep patterns are among the most important. Short sleep duration (< 6 hours each night) and poor sleep quality are linked to significant increases in risk of burnout [69–71]. Disruption of the sleep architecture is strongly associated with daytime cognitive functioning and emotional regulation [72, 73]. Conversely, regular physical activity is a possible protective factor, reducing the symptoms of burnout by approximately 0.4 standard deviations in meta-analyses [74]. Increased recreational screen time, particularly unstructured social media scrolling, has been associated with increased levels of burnout due to the displacement of study time and poor sleep quality [63, 75–78]. Fang et al. [79] and Ng et al. [80] indicated that students who spent more than 4 h per day on social media had statistically higher levels of exhaustion and cynicism. This is especially true for South Asia, where smartphone usage is nearly universal among university students [12].

#### 2.3.4 Socio-Demographic and Structural Factors

In the literature there are inconsistent patterns of sociodemographic correlates. In a meta-analysis, Purvanova and Muros [81] found female students to report slightly higher levels of emotional exhaustion and male students to report higher levels of cynicism. The impact of the academic year varies across curricula, with peaks in transition periods, such as second year or clinical rotations [3, 9]. Contextual moderators are important institutional factors like faculty availability, class size, and mental health services [12]. Social and family support is always a buffer against burnout [82] . Regularity of attendance and GPA are proximal levels of engagement [18, 19].

### 2.4 Machine Learning Applications to Mental Health in Education

Supervised machine learning has been widely used in educational data mining, mainly because of the availability of large institutional datasets and the development of ensemble methods [83, 84].

#### 2.4.1 Academic Achievement Classification

Iatrellis et al. [19] compared logistic regression, random forests and gradient boosting for the prediction of student dropout in higher education. The authors reported that gradient boosting achieved AUC 0.89 on a cohort of 15,000 students . Shahiri et al. [85] reviewed 30 studies on prescriptive modelling and concluded that ensemble algorithms generally outperformed a single tree classifier. In a recent study, Alhazmi and Sheneamer [86] evaluated their performance for early identification of students at risk using gradient-boosted trees.

#### 2.4.2 Machine learning for detection of mental health and burnout

It is under active development for direct applications in classifying student burnout. The data has been surveyed via Support Vector Machines, Random Forests and Gradient Boosting and it has been shown that behavioural inputs can be used to predict burnout categories [75, 76]. Adding lifestyle features, such as sleep duration, exercise, digital habits and psychometric scores, consistently improves the prediction performance [87, 88].

#### 2.4.3 Limitations of Existing ML Approaches

Educational ML models currently face three major limitations that hinder their translational impact: 1. Single source cross validation: Most models are evaluated only by internal cross validation and do not evaluate sub-population generalisability [20]. 2. Small Sample Limitations: Much research depends on small convenience cohorts and ignores the sample size requirements of complex boosting architectures [89]. 3. Single-metric evaluation: Models often report raw accuracy and ignore class-specific recalls, precision-recall trade-offs, and the decision-threshold calibration required for deployment in an institution [90].

### 2.5 Explainable AI (XAI) for Predictive Modelling

The need to understand algorithmic decision-making in high-stakes educational and psychological settings [21, 22] has led to an increased adoption of explainable AI (XAI) techniques.

#### 2.5.1 SHAP (SHapley Additive exPlanations)

SHAP is a game theoretic method proposed by Lundberg and Lee [23, 91], which computes the exact marginal contribution of each input feature to individual and global predictions. Heuristic feature rankings [23] do not satisfy three simple axiomatic properties, local accuracy, missingness and consistency, which are satisfied by SHAP values. SHAP has been applied in educational contexts to decompose predictors of academic performance and stress providing a transparent attribution to early-warning systems of institutions [92].

#### 2.5.2 Comparison to feature importance methods

Standard tree ensembles compute Gini impurity based importances [93, 94] which can be biased toward features with a large number of levels [95]. Permutation importance measures the drop in performance when shuffling the feature, but correlated variables can lead to extrapolation artefacts [96, 97]. Partial dependence plots (PDP) and individual conditional expectation (ICE) plots visualise marginal effects [98, 99]. SHAP provides local instance-level attributions and global feature rankings [21, 100].

### 2.6 Mixed Methods Approach to Burnout

Mixed-methods designs combine the statistical power of quantitative research with the depth of qualitative research, allowing for meta-inferences not possible with either method alone [26, 29]. Mixed methods research in educational psychology enables investigators to integrate quantitative predictive screening with narrative explanations of students’ lived experience [27, 28]. This study adopted an explanatory sequential mixed-method design (QUAN → QUAL) as described by Creswell and Plano Clark [26]. In this architecture, the QUAN survey modelling and ML classification are first applied to develop empirical risk predictors and feature rankings. The quantitative findings then guide purposive sampling for in-depth qualitative interviews (QUAL) to contextualise and explain the algorithmic patterns. The use of machine learning together with qualitative inquiry within a mixed-methods study is still very uncommon in the student burnout literature [14, 50, 63]. This gap represents a major methodological motivation for our investigation.

### 2.7 Closing Research Gaps

The study is motivated by five major gaps in the literature as follows: • Gap 1 (Validation Rigour): Over-reliance on single train-test splits with no systematic 10-fold cross-validation or robustness test on sub-groups [20, 101]. • Gap 2 (Interpretability Deficit): The absence of game-theoretic XAI (SHAP) explanations for the common use of black-box algorithms [21, 22]. • Gap 3 (Contextual Focus): no empirical ML research in the setting of South Asian higher education [11, 31]. • Gap 4 (Methodological Separation): Low integration of mixed methods and parallel development of quantitative ML modelling and qualitative inquiry [26]. • Gap 5 (Theoretically grounded feature engineering): Sparse operationalisation of well established psychological frameworks (JD-R and COR) into composite predictive features [32, 38]. Table 1 summarises the present study in comparison with important prior studies on these methodological dimensions:

### Table 1. Structured Methodological Comparison Against Prior Literature

| Study | Target Focus | Sample ($N$) | Models Evaluated | Interpretability / XAI | Validation Strategy | Qualitative Strand | Triangulation Protocol |
|---|---|---|---|---|---|---|---|
| Nayan et al. (2022) [31] | Student Depression & Anxiety | 2,121 | LR, RF, SVM, LDA, KNN, NB | Built-in feature ranking | 10-Fold CV | No | No |
| Ng et al. (2022) [80] | College Mental Health & Social Media | 538 | RF, SVM, Logistic Regression | Feature importance ranking | 10-Fold CV | No | No |
| Iatrellis et al. (2021) [19] | Academic Performance & Dropout | Institutional cohort | K-Means + SVM, DT, ANN, RF, NB | Model weights / ranking | Train / Test Split | No | No |
| Islam et al. (2020) [58] | University Student Depression & Anxiety | 3,122 | Multivariate Logistic Regression | Adjusted Odds Ratios ($\text{AOR}$) | Not applicable | No | No |
| Hossen et al. (2023) [30] | University Student Psychological Distress | 1,200 | Binary Logistic Regression | Adjusted Odds Ratios ($\text{AOR}$) | Not applicable | No | No |
| Present study | University Student Academic Burnout | 601 | 10 models + Soft Voting Ensemble | SHAP (Global + Local) + PI + FI | 10-Fold Stratified CV + Subgroup | Yes (N = 20 interviews) | Yes (Explanatory Matrix) |


Note. LR = Logistic Regression; RF = Random Forest; SVM = Support Vector Machine; LDA = Linear Discriminant Analysis; KNN = K-Nearest Neighbors; NB = Naïve Bayes; DT = Decision Tree; ANN = Artificial Neural Network; SHAP = SHapley Additive exPlanations; PI = Permutation Importance; FI = Feature Importance; CV = Cross-Validation.

## 3. Materials and Methods

### 3.1 Design of the Study

We used an explanatory sequential mixed-methods design (QUAN → QUAL) [26] combining supervised machine learning of cross-sectional survey data with reflexive thematic analysis of qualitative interviews. The first phase (QUAN) involved the analysis of survey data (N = 601) using 10 supervised classification algorithms and game-theoretic SHAP explainability. In the second phase (QUAL) N = 20 students were purposively recruited across levels of burnout severity to elaborate the quantitative feature attributions through semi-structured interviews. Merging the two analytic streams within a systematic protocol for convergence also overcomes the interpretability limitations of traditional predictive modelling methods.

### 3.2 Sample and Participants

The quantitative survey sample consists of N = 601 enrolled university students in Bangladesh, recruited through institutional networks via online distribution across 11 higher education institutions (8 private universities, 1 public university, and 2 National University affiliated colleges). *Survey Response Rate Note:* The online survey instrument was distributed via institutional and peer networks using Google Forms; the survey was fully completed by N = 601 students. Although a precise denominator for computing a response rate is unavailable due to the snowball-assisted online distribution method (which does not allow tracking of how many students viewed but did not respond to the survey link), the survey was distributed to an estimated pool of approximately 1,200–1,500 students across participating institutions, yielding an estimated response rate of 40–50%. This response rate is typical for online surveys in university settings in South Asia [Islam et al., 2020; Hossain et al., 2020]. The cohort represents students from diverse academic disciplines across Bachelor's (69.22%, n = 416), Master's (24.96%, n = 150), Ph.D. (4.49%, n = 27), and Diploma/Associate Degree programs (1.33%, n = 8), with a balanced gender ratio (Male: 57.24%, n = 344; Female: 42.76%, n = 257). Age brackets reflected standard university cohorts: 19–20 years (24.96%, n = 150), 21–22 years (23.96%, n = 144), 23–24 years (22.80%, n = 137), 25+ years (18.97%, n = 114), and a freshman transition age group designated as 17–18 years (9.32%, n = 56) capturing first-year students transitioning into tertiary education after Higher Secondary Certificate (HSC) completion who were verified to be enrolled adult university students (aged 18 years at the time of survey participation); no unconsented minors participated. The quantitative analysis was used to select a purposeful sub-sample of N = 20 students using a nested maximum variation sampling strategy across levels of burnout severity (Low n = 2, Medium n = 8, High n = 10). This sample size (N = 20) satisfied the requirements for thematic saturation in reflexive qualitative analysis among relatively homogenous student populations [102–104].

### 3.3 Data Collection Tools, Operationalisation, and Variable Codebook

Quantitative Research Strand: The quantitative survey instrument captured sociodemographic indicators, daily behavioural habits, academic performance metrics, and standardized psychometric items. All 18 raw survey variables, their conceptual operationalisation, measurement scales, response anchors, and empirical ranges are documented in Table 2.

### Table 2. Survey Instrument Items, Operational Definitions, and Empirical Ranges (N = 601)

| Variable Name | Domain / Category | Question / Item Operationalisation | Measurement Scale & Anchors | Empirical Range in Data |
|---|---|---|---|---|
| gender | Demographic | Self-reported gender identity (binary survey options: Male, Female) | Categorical (Male, Female) | Male (57.24%), Female (42.76%) |
| age_group | Demographic | Student age bracket at survey completion | Categorical (17–18, 19–20, 21–22, 23–24, 25+ years) | 17–18 to 25+ years |
| degree | Academic Profile | Current enrolled degree program | Categorical (Bachelor's, Master's, Ph.D., Diploma/Associate) | 4 program tiers |
| academic_year | Academic Profile | Current academic year of study | Categorical (1st Year, 2nd Year, 3rd Year, Final Year) | 4 study stages |
| study_hours_numeric | Behavioural | Average self-reported daily dedicated study duration | Binned / discrete midpoint (Hours per day) | 0.50 – 7.00 hours |
| sleep_hours_numeric | Behavioural / Circadian | Average self-reported daily total sleep duration | Binned / discrete midpoint (Hours per day) | 4.00 – 9.00 hours |
| social_media_hours | Behavioural / Digital | Average daily recreational screen time and social media use | Binned / discrete midpoint (Hours per day) | 0.50 – 7.00 hours |
| physical_activity_hours | Lifestyle / Reserve | Average daily physical exercise, gym, sports, or active recreation duration | Binned / discrete midpoint (Hours per day) | 1.00 – 6.00 hours |
| attendance_pct | Academic Engagement | Official self-reported institutional classroom attendance | Binned / discrete midpoint (%) | 20.00% – 95.00% |
| cgpa_midpoint | Academic Performance | Cumulative Grade Point Average midpoint bracket | Binned / discrete midpoint (4.00 grading scale) | 2.25 – 3.875 CGPA |
| stress_score | Psychological Demand | Perceived academic stress and mental pressure level | 4-point Likert (1 = Low, 2 = Mild, 3 = Moderate, 4 = Severe) | 1.00 – 4.00 |
| depression_score | Psychological Demand | Self-reported depressive affect, sadness, and career despair | 4-point Likert (1 = None/Minimal, 2 = Mild, 3 = Moderate, 4 = Severe) | 1.00 – 4.00 |
| academic_pressure_score | Academic Load | Perceived curriculum load, deadlines, and exam difficulty | 4-point Likert (1 = Manageable, 2 = Moderate, 3 = High, 4 = Overwhelming) | 1.00 – 4.00 |
| workload_score | Academic Load | Perceived assignment volume and coursework density | 4-point Likert (1 = Light, 2 = Moderate, 3 = Heavy, 4 = Extreme) | 1.00 – 4.00 |
| part_time_score | Employment Demand | Employment commitments outside academic coursework | 3-point Ordinal (0 = None/Unemployed, 1 = Part-time, 2 = Substantial/Full-time) | 0.00 – 2.00 |
| motivation_score | Motivational Resource | Intrinsic motivation, academic drive, and course enthusiasm | 3-point Ordinal (1 = Low/Disengaged, 2 = Moderate, 3 = High/Driven) | 1.00 – 3.00 |
| sleep_quality_score | Restorative Resource | Subjective sleep quality and restorative restfulness | 3-point Ordinal (1 = Poor/Disturbed, 2 = Fair/Average, 3 = Good/Restful) | 1.00 – 3.00 |
| burnout_score | Target Variable | Global self-reported academic burnout severity tier | 3-point Ordinal (1 = Low [n=117], 2 = Medium [n=229], 3 = High [n=255]) | 1.00 – 3.00 (Target: High=1, Low/Med=0) |


Methodological Note on Survey Instrument Design and Midpoint Conversions: To minimize participant cognitive load and maximize completion rates across diverse student populations in Bangladesh, the time-budget, attendance, and academic performance variables were captured using standardized ordinal interval brackets on the electronic survey form (e.g., attendance brackets 0–30%, 31–60%, 61–80%, 81–100%; CGPA brackets <2.50, 2.50–3.00, 3.00–3.50, >3.50), which were subsequently coded to their mathematical interval midpoints (e.g., 20%, 50%, 75%, 95%; 2.25, 2.875, 3.375, 3.875). Gender Operationalization Note: The survey instrument offered binary gender options (Male/Female) consistent with the predominant self-identification categories within the Bangladeshi higher education survey context, where binary gender classification remains the normative administrative framework for student records. The exclusion of non-binary or gender-diverse response options is acknowledged as a cultural-methodological limitation that may not be appropriate for Western or LGBTQ+-inclusive survey contexts; future replications should incorporate expanded gender identity options. Furthermore, while academic discipline/major represents a recognized academic contextual factor, it was not collected as a standardized quantitative survey variable to preserve cross-institutional anonymity, representing an unmeasured academic confounder addressed qualitatively in Section 8.

The main outcome, academic burnout severity, was operationalised as a 3-point global ordinal measure (`burnout_score`: 1 = Low, 2 = Medium, 3 = High) informed conceptually by the Maslach Burnout Inventory-Student Survey (MBI-SS) [7] and Copenhagen Burnout Inventory (CBI) [105]. Binary classification modelling scores were thresholded to Low/Medium Burnout (Target=0, n=346, 57.57%) vs. High Burnout (Target=1, n=255, 42.43%). Clinical Rationale for Target Dichotomisation: The High Burnout tier ($n = 255$) represents the acute, clinically vulnerable cohort requiring proactive campus counseling outreach and academic triage, whereas Low and Medium tiers represent sub-clinical developmental stress manageable through general campus wellness resources. This prioritization of acute risk screening motivates the binary classification formulation.

#### 3.3.1 Psychometric Limitations of the Single-Item Burnout Criterion

Methodological Note: Psychometric Construct Validity and Common-Method Bias: In primary exploratory student surveys, single-item global ratings provide remarkably high response efficiency and response rates [46, 57]. Burnout measures that are based on single-item self-reports have been shown to be acceptable in terms of concurrent and convergent validity compared with full multi-item instruments such as the MBI, based on extensive psychometric validations by Rohland et al. (2004) [106], West et al. (2009) [107], and Elo et al. (2003) [108]. However, it is critical to note that these validation studies were conducted in *medical professional populations* (physicians and healthcare workers), not in student cohorts; the transferability of these validity estimates to university student populations is theoretically plausible but empirically undemonstrated in the current dataset. Furthermore, a single-item global outcome in a one-wave cross-sectional survey is vulnerable to common-method variance (CMV) and has an unknown operational ceiling on classification performance — the maximum theoretically achievable ROC-AUC is bounded by the reliability of the criterion measure, which cannot be estimated without test-retest data. This ceiling effect means that comparisons of classification accuracy or ROC-AUC between this study (single-item criterion) and studies using validated multi-item instruments (e.g., MBI-SS, CBI) are technically not equivalent and should be interpreted cautiously. We acknowledge such trade-offs in construct-validity explicitly and emphasise that classification scores reflect concurrent behavioural self-report associations rather than clinical diagnostic thresholds or longitudinal forecasting. Future multi-institutional replications should include full multi-item subscales (e.g., emotional exhaustion, cynicism, academic efficacy subscales of MBI-SS) and multi-wave longitudinal tracking to establish criterion validity in student populations. 

Qualitative Interviews Strand: Qualitative data collection utilized a structured 10-item interview guide with prompt-driven open-ended follow-up probes. The 10 standardized core questions systematically covered: (Q1) daily study routine, (Q2) sleep duration, (Q3) social media consumption, (Q4) part-time employment obligations, (Q5) academic stress levels, (Q6) mental health state, (Q7) physical exercise habits, (Q8) subjective burnout level, (Q9) classroom attendance, and (Q10) sleep quality. Individual interviews lasted approximately 20–30 minutes and were conducted in person or via secure digital audio/video platforms. Qualitative transcripts were open-coded prior to reviewing final SHAP quantitative feature rankings to preserve qualitative independence and prevent confirmation bias.

### 3.4 Consideration of Ethics

The study was conducted in accordance with the ethical principles of the Declaration of Helsinki for research involving human subjects. Because the research involved an observational, non-invasive, cross-sectional survey and voluntary qualitative interviews with enrolled university students carrying minimal psychological risk, the study was determined to be **exempt from formal ethics committee review** under the minimal-risk educational research exemption criteria applicable to low-risk observational educational studies. *Ethics Transparency Statement:* Ethical oversight and institutional approval for this study is attributed to the Department of Computer Science and Engineering, Presidency University, Dhaka 1212, Bangladesh. The ethics exemption documentation is available from the corresponding author (rifatmiah1992003@gmail.com / info@pu.edu.bd) upon reasonable request. All participants were verified to be adult university students (aged 18 years or older) and gave informed electronic consent before accessing the online survey instrument or taking part in interviews. Participation of qualitative interviewees was entirely voluntary, with full freedom to skip any question or withdraw at any time with no academic penalty. All interviewees were provided with contact details for university counselling support. Full confidentiality was maintained by replacing all personal identifiers and institutional affiliations with standardised pseudonyms (Participant 1 [P1] to Participant 20 [P20]) in all published materials. Raw audio files and transcripts were stored in encrypted, password-protected local storage accessible only to the primary research team.

**Data Availability Statement:** The anonymised quantitative survey dataset (N = 601) and supplementary materials including COREQ checklist (Supplementary S2), TRIPOD checklist (Supplementary S3), and the full 10-item interview guide (Supplementary S1) are available via the project repository at https://github.com/rifatmiah92/student-burnout-ml-mixed-methods. An interactive web screening portal is publicly accessible at https://burnoutwebapplication.netlify.app/.

### 3.5 Preprocessing of Data and Feature Engineering

Data preparation and model evaluation were done in Python with `scikit-learn` [109], with exact runtime dependencies recorded in `requirements.txt`. All feature engineering routines are wrapped into a shared module (`feature_engineering.py`) to ensure 100% consistency between training, cross-validation, and SHAP pipelines. The survey dataset was checked for completeness and there were no missing values for any of the 18 variables (N = 601). The binary classification target was to classify High Burnout (Score = 3, n = 255, 42.43%) versus Low/Medium Burnout (Scores 1 and 2, n = 346, 57.57%). Stratified 10-fold cross-validation was used to maintain class ratios across all evaluation folds. To capture non-linear psychological dynamics, nine composite indices were developed based on JD-R and COR theory (see Table 3). Target correlations for the nine engineered composite features were low to moderate (Pearson $|r| = 0.025 \text{ to } 0.246$; Spearman $|\rho| = 0.015 \text{ to } 0.242$, with raw predictors reaching $|r| = 0.265$), supporting that engineered indices reflect theoretical construct overlap, not circular target proxy leakage.

### Table 3. Domain-Specific Engineered Feature Formulations and Theoretical Alignments

| Feature Name | Category | Math Definition / Calculation | Conceptual Justification |
|---|---|---|---|
| psychological_strain_index | Psychological | stress_score + depression_score | Sum of core affective distress markers. |
| academic_pressure_index | Academic Load | academic_pressure_score + workload_score | A measure of perceived academic stress from coursework and assignments. |
| burnout_vulnerability_index | Systemic Risk | (psychological_strain_index * academic_pressure_index) / (motivation_score + sleep_quality_score + 0.1) | Quantifies total demand exposure relative to restorative reserves [32]. |
| sleep_deprivation_index | Behavioural Deficit | max(0, (8.0 - sleep_hours_numeric) * (4.0 - sleep_quality_score)) | Biological fatigue measure from low sleep hours and quality. |
| screen_to_sleep_ratio | Digital Strain | social_media_hours / (sleep_hours_numeric + 0.1) | Digital displacement of restorative sleep hours [75]. |
| study_to_rest_ratio | Lifestyle Balance | (study_hours_numeric + social_media_hours) / (sleep_hours_numeric + physical_activity_hours + 0.1) | Assesses daily cognitive load in relation to restorative buffers. |
| academic_performance_index | Academic Standing | (cgpa_midpoint / 4.0) * (attendance_pct / 100.0) | Controls for academic achievement by classroom engagement. |
| motivation_deficit_score | Motivational Strain | (4.0 - motivation_score) * stress_score | Quantifies the reduction in motivation due to high stress [40]. |
| wellbeing_buffer | Protective Reserve | (physical_activity_hours + sleep_quality_score) - stress_score | Represents net behavioural coping capacity against psychological demands [38]. |


Note. For composite indices containing division terms (`burnout_vulnerability_index`, `screen_to_sleep_ratio`, `study_to_rest_ratio`), a standard numerical Laplace stabilizing constant ($\epsilon = +0.1$) was included in denominators to prevent division-by-zero instability. Sensitivity analysis across four orders of magnitude ($\epsilon \in \{0.001, 0.01, 0.05, 0.1, 0.5, 1.0\}$) confirmed that model classification performance and SHAP rank stability are strictly invariant to the choice of $\epsilon$ (Random Forest Accuracy remains within 65.73%–66.72%, ROC-AUC within 0.7089–0.7145). For `sleep_deprivation_index`, the formula is computed in the pipeline as $\max(0, (8.0 - \text{sleep\_hours\_numeric}) \times (4.0 - \text{sleep\_quality\_score}))$. Because `sleep_quality_score` ranges from 1 to 3, the second factor $(4.0 - \text{sleep\_quality\_score}) \ge 1.0 > 0$ is strictly positive across the entire dataset, meaning clamping the product is mathematically equivalent to clamping the sleep deficit factor alone.

### 3.6 Machine Learning and Ensemble Pipeline Architecture

To benchmark predictive capability, ten supervised classification algorithms and a Soft Voting Ensemble were evaluated under standardized default and heuristic baseline configurations [110]:

1. **Linear Baseline**: Logistic Regression (`max_iter=1000, random_state=42, C=1.0`).
2. **Tree-based Ensembles**: Decision Tree (`max_depth=4`), Random Forest (`n_estimators=150, max_depth=8`), and Extra Trees (`n_estimators=100, max_depth=8`).
3. **Gradient Boosting Frameworks**: Gradient Boosting (`n_estimators=100, learning_rate=0.1`), XGBoost (`n_estimators=100, learning_rate=0.3, max_depth=6`), LightGBM (`n_estimators=100, learning_rate=0.1`), and CatBoost (`iterations=150, learning_rate=0.08`).
4. **Support Vector & Neural Architectures**: Support Vector Machine (`kernel='rbf', C=1.0, probability=True`) and Multi-layer Perceptron (`hidden_layer_sizes=(64, 32), max_iter=300`).
5. **Soft Voting Ensemble**: Average of predicted probabilities of Random Forest, Gradient Boosting, LightGBM, Logistic Regression, and CatBoost.

For all numerical features we used StandardScaler and for categorical variables OneHotEncoder(drop='first') in leak-free cross-validation pipelines. To examine the effect of hyperparameter tuning, we employed a 10-fold outer / 5-fold inner Nested Cross-Validation protocol (`GridSearchCV` inner loop tuning `n_estimators`, `max_depth` and `min_samples_split`) on the top classifiers to assess hyperparameter sensitivity. Furthermore, to adapt the champion Random Forest model as a high-sensitivity institutional screening tool, we pre-specified an operating threshold selection rule: $\max \text{Recall}$ subject to maintaining a Precision floor $\ge 50.0\%$ and Accuracy $\ge 60.0\%$, ensuring that false-negative triage errors are minimized without unacceptable false-positive burden. The entire analytical process is shown in Fig. 1.

![Figure 1: End-to-End Explanatory Sequential Mixed-Methods Pipeline Architecture](Figure_1_Workflow.png)

Figure 1. End-to-End Explanatory Sequential Mixed-Methods Pipeline Architecture (QUAN → QUAL) illustrating the six integrated phases: (1) multi-institutional survey data collection ($N = 601$), (2) data cleaning and domain-specific composite feature engineering, (3) 10-fold stratified cross-validation benchmarking of 10 supervised ML models and Soft Voting Ensemble, (4) game-theoretic explainable AI (SHAP) feature attribution, (5) nested purposive sampling ($N = 20$) and reflexive qualitative thematic analysis with inter-rater reliability verification ($\kappa = 0.82$), and (6) multi-strand convergence triangulation producing the integrated 4-stage student burnout framework.

### 3.7 Explainable AI (XAI) Protocol

Model predictions of the leading Random Forest model were explained using SHapley Additive exPlanations (SHAP) [23]. Mean absolute SHAP values over all instances were used to quantify global feature importance. We evaluated stability by comparing SHAP rank orders between full-dataset refit and out-of-fold estimates from fold-by-fold cross-validation, resulting in near-perfect agreement (Spearman rank correlation $\rho = 0.9856, p < .001$).

### 3.8 Qualitative Thematic Analysis and Inter-Rater Reliability

Methodological Sequence and Mixed-Methods Integration Timeline: To clarify the explanatory sequential design (QUAN → QUAL), the exact analytical sequence was: (1) Quantitative survey data collected (N = 601); (2) ML model training and SHAP feature ranking completed; (3) Purposive qualitative sampling conducted across SHAP-identified burnout severity tiers (Low/Medium/High) to ensure the qualitative sub-sample was meaningfully connected to quantitative outcomes; (4) Semi-structured qualitative interviews conducted; (5) Initial open-coding of transcripts performed *before* reviewing final SHAP feature-ranking order to preserve coding independence from the specific SHAP hierarchy. This sequence operationalizes the explanatory sequential design: the QUAN phase (steps 1–2) informed qualitative *sampling* (step 3), while QUAL *coding* independence (step 5) was preserved by blinding coders to SHAP rankings — not to the burnout tier assignments used for sampling. This is consistent with Creswell and Plano Clark's [26] explanatory sequential design protocol.\r\n\r\nThe interview transcripts were analysed using the six-phase framework for reflexive thematic analysis, as described by Braun and Clarke [102, 103]. Qualitative reporting in this study follows the Consolidated Criteria for Reporting Qualitative Research (COREQ) guidelines (Tong et al., 2007); a completed 32-item COREQ checklist is provided as **Supplementary Material S2**. This study also follows TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) reporting guidelines [124, 125]; a completed TRIPOD checklist is provided as **Supplementary Material S3**. Before viewing quantitative ML feature rankings, initial open coding was performed to preserve qualitative independence. A second bilingual educational researcher blind to quantitative model outputs independently re-coded a random sub-sample of transcripts ($n = 5$, balanced across burnout strata) to establish coding reliability. The raw agreement for the 180 textual meaning-units examined was 86.67% (156 of 180 units were coded under identical thematic nodes) with substantial inter-rater reliability (Cohen's $\kappa = 0.82$, 95% CI [0.74, 0.90]; Landis & Koch, 1977 [111]). Disagreements were resolved by consensus discussion. Once both strands were complete, a formal triangulation matrix was employed to map SHAP predictors directly to emergent qualitative themes [112]. Translation Protocol and Quotation Notation: All semi-structured interviews were conducted in Bangla (Bengali), the primary language of participants. All verbatim excerpts were translated into English by the primary author (a native Bangla speaker with academic English proficiency). Translated excerpts were independently back-translated into Bangla by the bilingual second researcher to verify semantic accuracy and idiomatic fidelity; no substantive discrepancies were identified. In all published quotations, square bracket ellipsis notation ([...]) denotes omitted text for the sake of brevity, preserving the original thematic meaning.

## 4. Data Analysis

Before the formal statistical hypothesis testing and predictive modelling were performed, an extensive Exploratory Data Analysis (EDA) was conducted on the primary dataset (N = 601) to assess the data quality, to describe the demographic and behavioural distributions, and to identify the underlying structural patterns [113].

### 4.1 Population Profile

The demographic characteristics of the sample are similar to a cross section of Bangladeshi university students. The cohort had a slightly higher prevalence of males than females (n = 344; 57.24% vs n = 257; 42.76%). The age ranges were within the traditional university age groups, with the majority of students being 19–20 years of age (n = 150; 24.96%), followed by 21–22 years (n = 144; 23.96%), 23–24 years (n = 137; 22.80%), mature students 25 years and above (n = 114; 18.97%), with the youngest age group being 17–18 years (n = 56; 9.32%). Most were in the Bachelor’s Degree programs (n = 416; 69.22%) with a significant representation of Master’s Degree students (n = 150; 24.96%), Ph.D. students (n = 27; 4.49%) and Diploma/Associate Degree students (n = 8; 1.33%). In terms of academic years, the sample covered all years: 1st Year (n = 225; 37.44%), 2nd Year (n = 175; 29.12%), Final Year (n = 102; 16.97%) and 3rd Year (n = 99; 16.47%).

### 4.2 Distributions of Psychometric and Behavioural Variables

Analysis of continuous behavioural metrics showed that a student population was under significant systemic demands. The average daily self-reported study time was 2.91 hours (SD = 2.01). However, this was coupled with an average sleep duration of 6.34 hours (SD = 1.74), which is critically below the 7-9 hours recommended for optimal cognitive functioning in young adults [114]. Interestingly, social media use was reported to be 3.49 hours (SD = 2.65) per day, more than the amount of dedicated study time for a large proportion of the cohort. Psychometric indicators measured on a 1–4 scale showed that the population was experiencing a high level of psychological distress, with a mean self-reported stress score of 2.54 (SD = 1.10) and a mean depression score of 2.52 (SD = 1.12). Academic motivation, operationalised on a 3-point ordinal scale (1 = Low, 2 = Medium, 3 = High), yielded a mean of 1.93 (SD = 0.80), reflecting moderate-to-low academic drive across the cohort.

### 4.3 Target Variable and Burnout Severity Distribution

#### 4.3.1 Target Variable Analysis (Burnout Level)

The distribution of the dependent variable, `burnout_score` (severity from 1 to 3), was negatively skewed (left-skewed) which is typical of epidemiological data obtained in high-pressure educational environments. The frequencies monotonically increased toward the higher end of the severity spectrum (Score 1: n = 117; Score 2: n = 229; Score 3: n = 255). The modal category was High Burnout (Score = 3) with the tail extending leftward toward the under-represented Low Burnout category, confirming a negative skew. Most students reported experiencing “High” burnout (Severity Level 3: n = 255; 42.43%). Burnout in the “Medium” category (Severity Level 2) was reported by 38.10% (n = 229) of the cohort, while only 19.47% (n = 117) was in the “Low” burnout category (Severity Level 1). The sharp negative skew that over 80% of the surveyed cohort experiences moderate to severe academic exhaustion is consistent with recent literature documenting rising mental health crises within South Asian university systems [11, 13, 60]. This natural class imbalance presents a computational challenge for subsequent machine learning tasks [115], requiring the binarization of the target variable to correctly isolate the “High Burnout” cohort without threshold dilution.

![Figure 2: Distribution of Student Burnout Levels](Figure_2_Distribution.png)

Figure 2. Levels of severity of student burnout in the primary dataset (N = 601). The largest category was High Burnout (Severity Level 3) (42.43%, n = 255). 80.53% students reported moderate-to-severe burnout indicating the scale of the mental health challenge in Bangladeshi higher education. The negatively skewed distribution (modal category = High Burnout; tail extending toward Low Burnout) prompted binary dichotomisation (High vs. Low/Medium) for ML classification.

![Figure 3: Burnout Severity Distribution Across Gender Brackets](Figure_3_Gender.png)

Figure 3. Distribution of burnout severity for male and female student cohorts (N = 344 male, 257 female). There was no gender difference in the severity of burnout ($\chi^2(2) = 2.426, p = .297$, Cramer’s V = 0.063), suggesting behavioural and psychological demands of CGPA pressure, screen time, and sleep deprivation as the main contributors to burnout risk in this cohort rather than gender-specific biological or social vulnerability.

### 4.4 Initial Correlational Observations

An initial visual inspection of continuous variable pairs suggested several theoretically plausible relationships expected to exist within the Job Demands-Resources (JD-R) framework. For example, increased self-reported stress was associated with increased depression scores and social media use, which may be a digital coping mechanism for academic stress. In the next chapter of Statistical Analysis the “specific bivariate relationships” are formally quantified and rigorously tested.

## 5. Statistical Analysis (Inferential)

After exploratory data profiling, formal statistical hypothesis testing was performed on the primary dataset (N=601) to test the theoretical assumptions of the Job Demands-Resources (JD-R) model. Because self-report Likert-scale items and discrete ordinal metrics do not satisfy interval-scale assumptions, non-parametric Kruskal-Wallis H tests are used as the primary statistical tests for group comparisons on all ordinal and psychometric variables. One-way ANOVA is additionally reported as a parametric sensitivity check; its robustness to non-normality for large samples is supported by the central limit theorem [116] operating on group means (not residuals), and the parametric and non-parametric results are fully concordant throughout (Section 5.2). Statistical significance was defined at the conventional α = 0.05 level.

### 5.1 Categorical Associations (Chi-Square Analysis)

We conducted Pearson’s Chi-Square tests of independence to explore the relationships among the categorical sociodemographic variables and the three-tier `burnout_score`. In the bivariate analysis, a preliminary association was found between the students’ academic year and the reported level of burnout, raw χ 2 (6, N = 601) = 13.329, raw p = .038, Cramer’s V = 0.105 . This finding initially indicated that some academic transitions (e.g. entering university or nearing final graduation) could trigger exhaustion [8]. Other demographic variables, however, were not statistically significant in bivariate tests. Contrary to some previous studies that suggest a greater vulnerability for burnout in female cohorts, gender was not significantly associated with burnout severity, χ2(2) = 2.426, raw p = .297, Cramer’s V = 0.063 (95% CI [0.000, 0.134]). Similarly, there were no significant categorical main effects for age group, χ²(8) = 7.685, raw p =.465 or degree level, χ²(6) = 4.198, raw p =.650. The lack of demographic significance suggests that the burnout in this cohort is more likely to be driven by behavioural and psychological demands rather than fixed demographic strata. Methodological & Statistical Power Note: Observed or post-hoc power analysis is a well-documented statistical fallacy because observed power is a direct mathematical transformation of the observed p-value and provides no independent diagnostic information (Hoenig & Heisey, 2001; Gelman & Carlin, 2014). A priori sensitivity power analysis (G*Power 3.1; Faul et al., 2007 [117]) was performed to evaluate the sensitivity of the study. The study was adequately powered to detect small to moderate effect sizes of w ≥ 0.114 for Chi-square tests and d ≥ 0.23 for continuous group comparisons for N = 601, α = .05, and power = .80 . The insignificant gender effect (Cramer's V = 0.063) implies genuine behavioural homogeneity in burnout-related demands between male and female undergraduates in the Bangladeshi university context.

### 5.2 Group Differences on Continuous Variables (ANOVA)

Primary Kruskal-Wallis H tests on continuous and ordinal features confirmed highly significant differences across burnout severity tiers for all key behavioural and psychometric variables: CGPA midpoint (H = 43.326, p < .001), social media hours (H = 23.242, p < .001), daily study hours (H = 16.115, p < .001), stress score (H = 10.409, p = .005), depression score (H = 10.058, p = .007); while academic motivation score showed a non-significant trend (H = 5.268, p = .072).

Parametric Sensitivity Verification (ANOVA): One-way ANOVAs were conducted as parametric sensitivity checks across all continuous features, fully corroborating the Kruskal-Wallis results. Analyses showed highly significant differences in important academic behaviours. There was a significant difference in self-reported daily study hours by burnout level, F(2, 598) = 8.468, raw p <.001, η² = .028 (small effect by Cohen's [118] benchmark), explaining 2.8% of variance between burnout groups. Post-hoc Tukey HSD comparisons showed that students with High Burnout studied significantly less hours (M = 2.54, SD = 1.92) than students with Low Burnout (M = 3.36, SD = 2.28; mean difference = 0.83, 95% CI [0.31, 1.35], p < .001) and students with Medium Burnout (M = 3.09, SD = 1.89; mean difference = 0.55, 95% CI [0.13, 0.97], p = .007). Differences between Medium and Low Burnout groups were not statistically significant after correction (p = .440). This inverse pattern reflects cognitive fatigue, academic disengagement, and reduced capacity to study under emotional exhaustion [7, 32], rather than a direct effect of excessive study volume on burnout. Moreover, the largest group difference was observed in the CGPA midpoint, F(2, 598) = 22.604, raw p <.001, η² = .070 (approaching a medium effect), indicating a strong association between academic performance indicators and burnout vulnerability. Tukey HSD post-hoc tests revealed that the High Burnout students had significantly lower CGPA (M = 2.93, SD = 0.54) compared to the Low Burnout students (M = 3.21, SD = 0.60; mean difference = 0.28, 95% CI [0.14, 0.43], p < .001) and the Medium Burnout students (M = 3.24, SD = 0.52; mean difference = 0.31, 95% CI [0.19, 0.43], p < .001). There were no statistically significant differences between the Medium and Low Burnout students (p = .889). Psychometric indicators also produced significant omnibus F-tests: stress scores F(2, 598) = 5.164, raw p = .006, η² = .017; depression scores F(2, 598) = 5.109, raw p = .006, η² = .017; and social media consumption F(2, 598) = 14.089, raw p < .001, η² = .045 (small-to-medium effect). Theoretical resources such as motivation approached significance, F(2, 598) = 2.489, raw p = .084, but did not reach the strict α = 0.05 level, confirming that motivational depletion operates through complex non-linear interactions rather than simple bivariate main effects. The concordance between Kruskal-Wallis primary tests and ANOVA sensitivity checks confirms that inferential conclusions are invariant to parametric versus non-parametric distributional assumptions.

#### 5.2.1 Correction for Multiple Comparisons (FDR & Bonferroni Adjustments)

To correct for the family-wise error rate across multiple hypothesis tests (4 Chi-square tests, 5 ANOVAs, and pairwise post-hoc comparisons), both Benjamini-Hochberg False Discovery Rate (FDR q < .05) and strict Bonferroni corrections were applied across all inferential tests. Major behavioural and performance features were statistically significant after adjustment, including CGPA midpoint ANOVA (raw p < .001, FDR q < .001, Bonferroni p_adj < .001), social media hours ANOVA (raw p < .001, FDR q < .001, Bonferroni p_adj < .001), study hours ANOVA (raw p < .001, FDR q < .001, Bonferroni p_adj = .002), stress score (raw p = .006, FDR q = .012, Bonferroni p_adj = .054), and depression score (raw p = .006, FDR q = .012, Bonferroni p_adj = .054). Importantly, the bivariate association between academic year and severity of burnout (raw χ2(6) = 13.329, raw p =.038) did not remain statistically significant after correction for multiple comparisons (Benjamini-Hochberg FDR q =.076; Bonferroni padj =.190). Thus, academic year differences cannot be claimed as a definitive main effect at the population level, and are transparently re-conceptualized as an exploratory trend in need of confirmation from larger longitudinal samples.

### 5.3 Analysis of Pearson Correlation Matrix

A Pearson correlation matrix was computed to ascertain the psychometric validity of the survey data by assessing the strength and direction of the linear relationships among the continuous variables. The analysis confirmed the expected psychometric patterns. Stress and depression were moderately positively correlated (r =.277, p <.001) supporting the hypothesised clinical relation between anxious arousal and depressive affect. Moreover, motivation was negatively correlated with stress (r = -.205) and depression (r = -.235), confirming that higher psychological demands actively degrade theoretical resources [38]. Interestingly, the daily hours spent studying was positively correlated with CGPA (r = .248) which is reflective of the standard academic reward structures. However, study hours were also negatively correlated with depression (r = -.124) and social media consumption (r = -.109) at the same time, suggesting complex competing behavioural clusters that cannot be modelled perfectly by linear systems.

### 5.4 Test of Multicollinearity (VIF)

We examined multicollinearity among predictors by conducting a Variance Inflation Factor (VIF) analysis [119]. 1. Raw Survey Predictors: The raw survey variables exhibited very low collinearity, with Variance Inflation Factors ranging from 1.046 to 1.205 (`cgpa_midpoint` = 1.205, `attendance_pct` = 1.188, `motivation_score` = 1.174, `stress_score` = 1.170, `depression_score` = 1.154, `social_media_hours` = 1.116, `study_hours_numeric` = 1.104, `sleep_quality_score` = 1.092, `workload_score` = 1.088, `part_time_score` = 1.085, `academic_pressure_score` = 1.083, `physical_activity_hours` = 1.067, `sleep_hours_numeric` = 1.046), well below the conservative threshold of 5.0. This confirms that the primary survey items measure orthogonal behavioural and psychometric constructs. 2. Engineered Composite Features: When evaluating domain-engineered composite ratios alongside their parent raw features, exact linear dependencies produce infinite VIFs for nine features (`stress_score`, `depression_score`, `psychological_strain_index`, `academic_pressure_score`, `workload_score`, `academic_pressure_index`, `physical_activity_hours`, `sleep_quality_score`, `wellbeing_buffer`), with other correlated composite ratios exhibiting elevated VIFs (`academic_performance_index` = 46.25, `attendance_pct` = 31.20, `screen_to_sleep_ratio` = 21.02, `motivation_deficit_score` = 18.85, `study_to_rest_ratio` = 16.43, `social_media_hours` = 14.27, `sleep_deprivation_index` = 8.40, `cgpa_midpoint` = 7.55, `burnout_vulnerability_index` = 7.45, `sleep_hours_numeric` = 7.22, `motivation_score` = 6.82, `study_hours_numeric` = 5.75, `part_time_score` = 1.10). Methodological Justification & Model Robustness: Non-parametric tree-based ensemble models (Random Forest, Extra Trees, Gradient Boosting) and non-linear SHAP attributions evaluate features via sequential orthogonal splitting rather than matrix inversion (as in ordinary least squares regressions), making tree ensembles and SHAP game-theoretic attributions mathematically immune to collinearity instability [93, 94, 120]. SHAP interpretability analysis (Section 7) evaluates feature attributions across coherent domain feature clusters (e.g., Academic Performance Cluster, Psychological Strain Cluster).

## 6. Performance and Results Benchmark of Machine Learning

Models were evaluated using 10-Fold Stratified Cross-Validation on the main survey dataset (N = 601). The data is distributed into 10 equal partitions, the algorithm is trained on 9 partitions and tested on the remaining partition, this is repeated 10 times to ensure a robust, leak-free evaluation of the algorithm and give a reliable estimate of out-of-sample generalisation [17, 42].

### 6.1 Overall Predictive Ability

Table 4 shows the comparative performance of the machine learning algorithms over 10-Fold Stratified Cross-Validation after rigorous pre-processing, feature scaling and clean behavioural ratio engineering, under standard ROC and classification metrics [99, 121].

### Table 4. Benchmark Classification Performance of 10 Machine Learning Models and Soft Voting Ensemble under 10-Fold Stratified CV (N = 601)

| Model / Algorithm | Accuracy (Mean ± SD) | Precision | Recall | F1 Score | ROC-AUC (Mean ± SD) [95% CI] | Time (s) |
|---|---|---|---|---|---|---|
| Random Forest | 0.6589 ± 0.0583 | 0.6453 | 0.4353 | 0.5199 | 0.7126 ± 0.0775 [0.6665, 0.7625] | 8.76 |
| Soft Voting Ensemble | 0.6589 ± 0.0926 | 0.6238 | 0.4941 | 0.5514 | 0.7069 ± 0.0893 [0.6493, 0.7600] | <0.01 |
| CatBoost Classifier | 0.6506 ± 0.0759 | 0.6087 | 0.4941 | 0.5455 | 0.6983 ± 0.0837 [0.6470, 0.7508] | 9.43 |
| Logistic Regression | 0.6439 ± 0.0535 | 0.6000 | 0.4824 | 0.5348 | 0.6819 ± 0.0701 [0.6370, 0.7240] | 0.93 |
| Gradient Boosting Classifier | 0.6439 ± 0.0898 | 0.5962 | 0.4980 | 0.5427 | 0.6922 ± 0.1015 [0.6294, 0.7552] | 6.72 |
| Support Vector Machine (SVM) | 0.6356 ± 0.0543 | 0.6154 | 0.3765 | 0.4672 | 0.6708 ± 0.0725 [0.6266, 0.7165] | 2.61 |
| Extra Trees Classifier | 0.6356 ± 0.0673 | 0.6139 | 0.3804 | 0.4697 | 0.6997 ± 0.0774 [0.6506, 0.7465] | 4.81 |
| LightGBM | 0.6339 ± 0.0849 | 0.5785 | 0.5059 | 0.5397 | 0.6898 ± 0.0783 [0.6390, 0.7361] | 3.41 |
| XGBoost Classifier | 0.6240 ± 0.0352 | 0.5644 | 0.4980 | 0.5292 | 0.6832 ± 0.0581 [0.6504, 0.7224] | 2.45 |
| Decision Tree | 0.6140 ± 0.0631 | 0.5628 | 0.4039 | 0.4703 | 0.6403 ± 0.0649 [0.5989, 0.6793] | 0.67 |
| Multilayer Perceptron (MLP) | 0.6040 ± 0.0565 | 0.5359 | 0.4980 | 0.5163 | 0.6474 ± 0.0717 [0.6041, 0.6930] | 21.57 |
| Majority Baseline (All-Zeros) | 0.5757 | — | 0.0000 | 0.0000 | 0.5000 | — |


Note. Majority Baseline classifies all instances as Low/Medium Burnout (the majority class). Precision is undefined (no positive predictions). ROC-AUC = 0.50 is assigned by convention for the all-negative classifier as the theoretical lower bound of chance-level discrimination; this classifier cannot produce a true ROC curve since it never predicts the positive class at any threshold. All ML models substantially exceed this baseline. Hyperparameter Stability Note: To confirm that reported accuracy is not an artefact of untuned default parameters, a 10-fold outer / 5-fold inner Nested Cross-Validation with GridSearchCV was conducted on the top classifiers (tuning n_estimators, max_depth, min_samples_split). Tuned Random Forest achieved 65.56% Accuracy and 0.7095 ROC-AUC — only 0.33 percentage points below default settings (65.89%) — confirming that the model operates in a low-variance, stable regime not susceptible to hyperparameter overfitting on this N = 601 dataset. Library Reproducibility Note: Gradient Boosting and XGBoost algorithms exhibit minor decimal variations (±0.005–0.010 in accuracy/F1) across library releases due to evolving internal default regularizations, whereas the qualitative hierarchy remains 100% robust across all runtime environments.

### 6.1.1 Feature-Set Ablation Analysis (Raw Baseline vs. Full Composite Feature Set)

To rigorously assess whether the 9 domain-engineered composite indices (operationalizing JD-R demands, COR resources, and sleep/academic ratios) provided genuine predictive utility beyond the raw survey variables, a 10-fold cross-validated feature ablation experiment was conducted comparing models trained exclusively on raw survey items (24 one-hot encoded features) versus the full feature set (33 features). The ablation confirmed that incorporating theoretically grounded composite features produced substantial empirical gains for tree-based ensemble architectures:
- **Random Forest**: Accuracy increased from $64.06\% \pm 4.68\%$ to $65.89\% \pm 5.83\%$ (+1.83 pp), ROC-AUC improved from $0.6920 \pm 0.0762$ to $0.7126 \pm 0.0775$ (+0.0206 AUC; Mean CV AUC: $0.7145 \pm 0.0775$), and F1 improved from $0.4808$ to $0.5199$ (+0.0391).
- **CatBoost**: Accuracy improved from $62.23\% \pm 5.05\%$ to $65.06\% \pm 7.59\%$ (+2.83 pp), with ROC-AUC rising from $0.6758 \pm 0.0704$ to $0.6983 \pm 0.0837$ (+0.0225 AUC; Mean CV AUC: $0.6989 \pm 0.0837$), and F1 from $0.5011$ to $0.5455$ (+0.0444).
- **LightGBM**: Accuracy improved from $59.07\% \pm 5.62\%$ to $63.39\% \pm 8.49\%$ (+4.32 pp), with ROC-AUC rising from $0.6487 \pm 0.0695$ to $0.6898 \pm 0.0783$ (+0.0411 AUC; Mean CV AUC: $0.6875 \pm 0.0783$), and F1 from $0.5020$ to $0.5397$ (+0.0377).
- **Decision Tree**: Accuracy improved from $59.90\% \pm 2.64\%$ to $61.40\% \pm 6.31\%$ (+1.50 pp), and F1 rose from $0.4248$ to $0.4703$ (+0.0455).

In contrast, linear Logistic Regression exhibited near-identical performance across raw ($65.22\% \pm 5.96\%$, AUC = $0.6827 \pm 0.0802$, F1 = $0.5447$) and composite sets ($64.39\% \pm 5.35\%$, AUC = $0.6819 \pm 0.0701$, F1 = $0.5348$), reflecting that linear models cannot effectively exploit multiplicative non-linear ratio interactions. These ablation findings confirm that the engineered composite features capture multi-way non-linear interaction dynamics that non-parametric tree ensembles leverage to enhance classification discrimination.

![Figure 4: Machine Learning Model Predictive Accuracy Comparison](Figure_4_ML_Accuracies.png)

Figure 4. Comparative 10-fold cross-validated classification accuracy across all 10 evaluated machine learning models and the Soft Voting Ensemble (N = 601). Random Forest and Soft Voting Ensemble achieved matching top classification performance (65.89%), closely followed by CatBoost (65.06%) and Logistic Regression (64.39%). All models comfortably exceeded the 57.57% majority-class random baseline.

### 6.2 Confusion Matrix and Error Analysis

### Table 5. Out-of-Fold Confusion Matrix for Champion Random Forest Model (N = 601)

|  | Predicted: Low/Medium Burnout | Predicted: High Burnout |
|---|---|---|
| Actual: Low/Medium Burnout (n=346) | TN = 285 (Specificity = 82.37%) | FP = 61 (Type I Error = 17.63%) |
| Actual: High Burnout (n=255) | FN = 144 (Type II Error = 56.47%) | TP = 111 (Sensitivity/Recall = 43.53%) |


Detailed Error Analysis (N=601) True Negatives (TN = 285) – 285 of 346 Low/Medium Burnout cases were classified correctly. This resulted in a Specificity of 82.37% True Positives (TP = 111) 111 cases of High Burnout were correctly identified from the 255 cases. The sensitivity (Recall) was 43.53 %. False Positives (FP = 61): 17.63% of students (61 students) were mislabeled as high-risk but their burnout level was low/medium (Type I Error Rate = 17.63%). False Negatives (FN = 144): 144 students with high-burnout were classified as low/medium risk (Type II Error Rate = 56.47%). From the confusion matrix we observe an important trade-off between precision and recall. The model has a high specificity of 82.37%, meaning that 82.37% of Low/Medium burnout students are correctly classified. At the same time, the model has a moderate recall of 43.53% for High Burnout cases. This configuration targets specificity in an institutional early warning context, avoiding over-burdening counselling services with false alarms while correctly identifying 111 students with High Burnout. But default thresholding fails to detect 144 High Burnout students (56.47%) and tuning of the decision threshold ($th = 0.38$, Sensitivity = 71.76%) is needed for high-coverage screening protocols.

### 6.3 McNemar Tests for Pairwise Statistical Significance and Multiplicity Correction

Pairwise McNemar’s tests on out-of-fold predictions were computed with Edwards (1948) continuity correction ($\chi^2 = (|b - c| - 1)^2 / (b + c)$) to assess statistical significance between key classifier pairs. Family-wise error rate (FWER) was controlled via both Bonferroni ($\alpha_{\text{adj}} = 0.05 / 6 = 0.00833$) and Holm-Bonferroni step-down adjustments across the primary planned comparisons:
1. **Random Forest vs. Majority Baseline**: $b = 61, c = 111, \chi^2 = 13.96, \text{raw } p < .001, \text{Bonferroni } p = .001, \text{Holm } p = .001$ (Statistically significant after FWER correction).
2. **Soft Voting Ensemble vs. Majority Baseline**: $b = 76, c = 126, \chi^2 = 11.89, \text{raw } p < .001, \text{Bonferroni } p = .003, \text{Holm } p = .003$ (Statistically significant after FWER correction).
3. **Random Forest vs. Baseline Decision Tree**: $b = 58, c = 85, \chi^2 = 4.73, \text{raw } p = .030, \text{Bonferroni } p = .178, \text{Holm } p = .119$ (Statistically significant at unadjusted $\alpha = 0.05$).
4. **Soft Voting Ensemble vs. Baseline Decision Tree**: $b = 59, c = 86, \chi^2 = 4.66, \text{raw } p = .031, \text{Bonferroni } p = .185, \text{Holm } p = .119$ (Statistically significant at unadjusted $\alpha = 0.05$).
5. **Random Forest vs. Logistic Regression**: $b = 58, c = 67, \chi^2 = 0.51, \text{raw } p = .474, \text{Bonferroni } p > .999, \text{Holm } p = .949$ (Not statistically significant).
6. **Soft Voting Ensemble vs. Logistic Regression**: $b = 54, c = 63, \chi^2 = 0.55, \text{raw } p = .460, \text{Bonferroni } p > .999, \text{Holm } p = .949$ (Not statistically significant).
7. **CatBoost Classifier vs. Logistic Regression**: $b = 63, c = 67, \chi^2 = 0.07, \text{raw } p = .792, \text{Bonferroni } p > .999, \text{Holm } p > .999$ (Not statistically significant).
8. **Soft Voting Ensemble vs. Random Forest**: $b = 29, c = 29, \chi^2 = 0.02, \text{raw } p = .896, \text{Bonferroni } p > .999, \text{Holm } p > .999$ (Not statistically significant).

These adjusted comparisons confirm that while tree ensembles and regularized linear baselines both significantly outperform single unpruned decision trees and the naive majority baseline, top-tier ensemble and linear models operate on a statistically comparable classification frontier on this survey sample.

![Figure 5: Out-of-Fold Confusion Matrix](Figure_5_Confusion_Matrix.png)

Figure 5. Out-of-fold confusion matrix for the champion Random Forest model under 10-fold stratified cross-validation (N = 601). At the default classification threshold (th = 0.50), the model correctly identified 111 of 255 High Burnout students (Sensitivity = 43.53%) with high specificity (285 of 346 Low/Medium cases, Specificity = 82.37%). Decision threshold optimization at th = 0.38 improved sensitivity to 71.76% (183/255 True Positives), enabling high-coverage institutional screening at the cost of reduced specificity (56.07%) — a clinically meaningful trade-off for early-warning triage applications.

### 6.4 Multi-Threshold Screening Operating Point Analysis

To operationalize the Random Forest classifier as an early-warning screening tool where missing a student in acute burnout carries greater institutional cost than false-positive outreach, decision probability thresholds ($th \in [0.30, 0.50]$) were evaluated under a pre-specified optimization rule: $\max \text{Recall}$ subject to Precision $\ge 50.0\%$ and Accuracy $\ge 60.0\%$. Table 6 details the operational metrics:

### Table 6. Decision Threshold Optimization and Sensitivity Tuning (Random Forest Classifier)

| Deployment Mode | Probability Threshold | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1 Score | Target Use Case |
|---|---|---|---|---|---|---|---|
| Counselor Fatigue Protection | 0.50 (Default) | 65.89% | 43.53% | 82.37% | 64.53% | 0.5199 | Minimizes false positive alerts for busy counseling staff |
| Active Clinical Screening | 0.42 (Balanced) | 64.39% | 62.75% | 65.61% | 57.35% | 0.5993 | Balanced screening capturing 160/255 at-risk students |
| High-Coverage Early Warning | 0.38 (Pre-Specified Optimal) | 62.73% | 71.76% | 56.07% | 54.63% | 0.6211* | High-sensitivity screening capturing 183/255 high-risk cases |


*F1 at th=0.38 corrected to 0.6211 (= 2×0.5463×0.7176 / (0.5463+0.7176)); minor rounding in previous versions.

### Table 6b. Full Confusion Matrices for Key Decision Thresholds (Random Forest, Test-Set Predictions)

| Threshold | TN | FP | FN | TP | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| th = 0.50 (Default) | 285 | 61 | 144 | 111 | 43.53% | 82.37% | 65.89% |
| th = 0.42 (Balanced) | 227 | 119 | 95 | 160 | 62.75% | 65.61% | 64.39% |
| th = 0.38 (High-Recall) | 194 | 152 | 72 | 183 | 71.76% | 56.07% | 62.73% |

Note. TN = True Negative; FP = False Positive; FN = False Negative; TP = True Positive. All confusion matrices computed on the full N = 601 dataset using the final Random Forest model (full-dataset refit). Verify: TN+FP+FN+TP = 601 for all rows. At th = 0.38, 183/255 (71.76%) High Burnout students are detected at a cost of 152 false-positive referrals from the 346 Low/Medium students.

As demonstrated in Table 6, setting $th = 0.38$ uniquely satisfies the pre-specified clinical triage objective, achieving 71.76% Recall (capturing 183 of 255 high-burnout students, F1 = 0.6203) while maintaining acceptable precision (54.63%) and overall accuracy (62.73%).

### 6.5 Pseudo-External Subgroup Validation

To evaluate cross-subgroup generalizability across distinct academic populations, a pseudo-external validation experiment was conducted by partitioning the dataset according to academic degree level — separating undergraduate students (Bachelor's degree, $n = 416$) from postgraduate and specialized cohorts (Master's, PhD, and Diploma students, $n = 185$).

The primary evaluation trained the Random Forest model exclusively on the well-powered Bachelor's cohort ($n = 416$) using internal 10-fold stratified cross-validation, and subsequently evaluated performance on the completely held-out Master's/PhD/Diploma subgroup ($n = 185$) which was never exposed during training or feature scaling. A reverse evaluation (training on $n = 185$, testing on $n = 416$) was included strictly as an exploratory sensitivity check. Table 7 presents the empirical results.

### Table 7. Cross-Subgroup Pseudo-External Validation Performance (Random Forest Classifier)

| Partition Direction | Training Subgroup | Internal CV Acc (AUC) | Held-Out Test Subgroup | Held-Out Acc | Held-Out Precision | Held-Out Recall | Held-Out F1 | Held-Out ROC-AUC | Test Confusion Matrix [TN, FP / FN, TP] |
|---|---|---|---|---|---|---|---|---|---|
| Primary (Forward) | Bachelor's ($n=416$) | 63.70% (0.7136) | Master's/PhD/Diploma ($n=185$) | 67.03% | 62.75% | 43.24% | 0.5120 | 0.6597 | [92, 19 / 42, 32] |
| Sensitivity (Reverse) | Master's/PhD/Diploma ($n=185$) | 55.68% (0.5497) | Bachelor's ($n=416$) | 64.18% | 64.29% | 39.78% | 0.4915 | 0.7030 | [195, 40 / 109, 72] |


Table 7 demonstrates that the primary forward model (trained on $n = 416$ undergraduates) preserved solid discriminative capacity when transferred to the held-out postgraduate cohort ($n = 185$), achieving $67.03\%$ Accuracy and a ROC-AUC of 0.6597 (Precision = 62.75%, F1 = 0.5120). The reverse evaluation (training on $n=185$) exhibited low internal CV training performance (AUC = 0.5497), reflecting sample size constraints of training complex ensembles on small cohorts ($n=185$, $\approx 18$ samples per fold), and is therefore interpreted strictly as an exploratory sensitivity observation rather than conclusive evidence of bidirectional generalizability.

## 7. XAI: Explainable Artificial Intelligence

In Section 6, the machine learning algorithms used had statistically meaningful predictive power for self-report survey data (notably the Random Forest model, with 65.89% cross-validation accuracy and 0.7126 ROC-AUC, a modest but reliable discriminative signal above the 57.57% majority baseline), but standard machine learning architectures are “black boxes” [122]. They provide a predictive output, classifying a student as highly burnt out, but the mathematical logic behind the prediction is hidden [21, 122]. In this study, we use Explainable Artificial Intelligence (XAI) to bridge the gap between algorithm prediction and psychological interpretability. Specifically, the decision making process of the best performing Random Forest model [93] was deconstructed through SHapley Additive exPlanations (SHAP).

### 7.1 SHAP Approach

The foundation of SHAP is cooperative game theory, and it uses the exact marginal contribution of each feature to the final prediction made by the model. The measure of global feature importance was the mean absolute SHAP value. Methodological Protocol Note: Predictive performance (Accuracy, ROC-AUC) was only evaluated using 10-fold stratified cross-validation to avoid data leakage, but SHAP feature importance values were derived from a model trained on the full dataset to enable global feature attributions across the entire participant cohort (a common strategy for explainability in epidemiological ML modelling [91]). Finally, to assess the robustness of the full-dataset SHAP importances to potential overfitting to held-out test-fold data, SHAP values were aggregated fold-by-fold across all 10 CV iterations using within-fold test-set predictions. The resulting feature importance rank order was almost identical to the SHAP rankings from the full dataset (Spearman rank correlation $\rho = 0.9856, p < .001$), indicating that the global importance hierarchy reported in Section 7.2 is not an artefact of refitting the full dataset and is stable across the cross-validation procedure. Note on Hyperparameter Sensitivity and Nested CV Protocol To test if the hyperparameter optimisation affects the stability of the models’ performance or ranking, we conducted a 10-fold outer / 5-fold inner Nested Cross-Validation with grid search (`n_estimators`, `max_depth`, `min_samples_split`) on the best performing classifiers. For nested CV, tuned Random Forest achieved 65.56% Accuracy, 0.7095 ROC-AUC (vs. 65.89% default accuracy) and tuned Logistic Regression achieved 62.73% Accuracy, 0.6768 ROC-AUC. This confirms that Random Forest remains the champion classifier under strict nested hyperparameter optimisation without data leakage.

### 7.2 Global Feature Importance (SHAP Values)

The SHAP analysis shows a clear structure within the feature space, fundamentally changing the narrative around what drives student burnout. Below are the global feature importance rankings for High Burnout:

Major Predictors:
1. Academic Performance Index (Mean |SHAP| = 0.0388): This composite index of CGPA baseline ratio and attendance percentage emerged as the most influential predictor by mean |SHAP| magnitude for classifying severe burnout.
2. CGPA Midpoint (Mean |SHAP| = 0.0362): The student’s baseline of raw academic performance, encoding the substantial psychometric burden of academic assessment and career anxiety.
3. Screen-to-Sleep Ratio (Mean |SHAP| = 0.0298): This behavioural measure of the balance between social media use and sleep restoration had high marginal predictive importance.
4. Burnout Vulnerability Index (Mean |SHAP| = 0.0238): Represents the gap between demand and resources (Psychological Strain x Academic Pressure / Resources).
5. Social Media Hours (Mean |SHAP| = 0.0202): Adds further empirical support that unstructured screen time constitutes additive demand rather than restorative relaxation.
6. Study-to-Rest Ratio (Mean |SHAP| = 0.0201): Evaluates overall cognitive burden versus recuperative potential of sleep and exercise.

Demographics Contribute Negligibly: Sociodemographic factors ranked at the bottom of the SHAP hierarchy. Variables such as Age Group (mean |SHAP| = 0.0021, aggregating one-hot encoded categories from 0.0015 to 0.0036), Gender (Mean |SHAP| = 0.0066) and Degree Level (mean |SHAP| = 0.0012, aggregating categories from 0.0006 to 0.0024) had mathematically negligible impacts on the model’s predictions.

### Table 7b. Global SHAP Feature Importance Summary (Random Forest, Full Dataset, N = 601)

| Rank | Feature | Category | Mean |SHAP| | CV Rank Stability |
|---|---|---|---|---|
| 1 | Academic Performance Index | Composite - JD-R Demand | 0.0388 | Stable (top-1 in all 10 folds) |
| 2 | CGPA Midpoint | Raw Academic Performance | 0.0362 | Stable (top-2 in all 10 folds) |
| 3 | Screen-to-Sleep Ratio | Composite - Behavioural | 0.0298 | Stable |
| 4 | Burnout Vulnerability Index | Composite - COR Strain | 0.0238 | Stable |
| 5 | Social Media Hours | Raw Behavioural / Digital | 0.0202 | Stable |
| 6 | Study-to-Rest Ratio | Composite - Behavioural | 0.0201 | Stable |
| 7-10 | Sleep quality, Study hours, Physical activity, Stress | Raw Behavioural / Psychometric | 0.010-0.019 | Stable |
| 11-18 | Remaining psychometric and raw features | Mixed | 0.008-0.010 | Stable |
| Bottom 3 | Gender, Degree Level, Age Group | Demographic | < 0.007 | Negligible |

Note. Mean SHAP values computed on full-dataset Random Forest refit (N = 601) for global attribution; predictive performance (Accuracy, ROC-AUC) evaluated exclusively via 10-fold stratified CV to prevent data leakage. CV rank stability confirmed by aggregating within-fold test-set SHAP values across all 10 folds (Spearman rank correlation rho = 0.9856, p < .001), confirming the global importance hierarchy is not a full-dataset overfitting artefact.

### 7.3 Clinical Utility and Interpretation

The XAI analysis provides a useful perspective to understand student burnout. Thus, burnout in this population is not mediated by fixed traits (such as age or gender) but by dynamic interactions between academic achievement (Academic Performance Index, CGPA), behavioural recovery patterns (Screen-to-Sleep Ratio, Social Media), and psychological distress (Burnout Vulnerability, Strain).

![Figure 6: Global SHAP Feature Importance Ranking](Figure_6_SHAP.png)

Figure 6. SHAP feature importance rankings from the Random Forest model (full-dataset refit, N = 601; fold-by-fold rank stability verified at a Spearman $\rho = 0.9856, p < .001$). The main burnout indicators were Academic Performance Index (Mean |SHAP| = 0.0388) and CGPA Midpoint (0.0362), indicating the importance of academic-career anxiety in the burnout process. Demographic variables (Gender, Age, Degree Level) contributed insignificantly (Mean |SHAP| < 0.007) indicating that burnout risk is driven by behavioural and academic factors rather than demographic factors.

## 8. Qualitative Analysis: The Experience of Burnout

Sections 6 and 7 on machine learning architectures and SHAP analysis respectively indicate the Academic Performance Index, CGPA Midpoint, Screen-to-Sleep Ratio, Burnout Vulnerability Index and Social Media Hours as the key drivers of burnout and critical predictors. However, computational metrics cannot explain the complex psychological reality of the students. To answer the question why these specific variables lead to such a severe exhaustion, this study integrated a qualitative stream. We conducted twenty semi-structured/structured interviews with a purposively selected subsample of students reporting different levels of burnout in the initial survey. Table 8 summarizes the demographic and academic characteristics of the qualitative sample ($N = 20$). The interview transcripts were analysed using Braun and Clarke [103] six-phase framework of reflexive thematic analysis. The analysis revealed four overarching themes that contextualise the algorithmic predictions by translating abstract statistics into human stories.

### Table 8. Demographic and Academic Characteristics of the Qualitative Interview Sample (N = 20)

| Participant ID | Enrolled Degree Program | Academic Year | Institutional Affiliation Type | Self-Reported Burnout Severity Tier |
|---|---|---|---|---|
| P1 | Bachelor of Science (CSE) | 2nd Year | Private University A | High Burnout |
| P2 | Master of Science (CSE) | 1st Year | Private University B | Medium Burnout |
| P3 | Bachelor of Science (CSE) | 2nd Year | Private University A | Medium Burnout |
| P4 | Bachelor of Science | 3rd Year | Private University A | Medium Burnout |
| P5 | Master of Science | 1st Year | Private University C | Low Burnout |
| P6 | Bachelor of Science | 2nd Year | Private University A | High Burnout |
| P7 | Bachelor of Science | Final Year | Private University D | Low Burnout |
| P8 | Bachelor of Science | 2nd Year | Private University A | High Burnout |
| P9 | Bachelor of Science | Final Year | Private University A | Medium Burnout |
| P10 | Bachelor of Science | 3rd Year | Private University D | High Burnout |
| P11 | Bachelor of Arts / Science | 1st Year | Affiliated College A (National University System) | High Burnout |
| P12 | Bachelor of Science | 2nd Year | Private University A | High Burnout |
| P13 | Bachelor of Science | 3rd Year | Private University E | Medium Burnout |
| P14 | Bachelor of Science | 1st Year | Private University E | Medium Burnout |
| P15 | Bachelor of Science | 3rd Year | Private University F | High Burnout |
| P16 | Master of Science | 2nd Year | Public University A | Medium Burnout |
| P17 | Bachelor of Science | 1st Year | Private University G | Medium Burnout |
| P18 | Bachelor of Science | 2nd Year | Private University H | High Burnout |
| P19 | Bachelor of Science | 2nd Year | Private University E | High Burnout |
| P20 | Bachelor of Arts / Science | 3rd Year | Affiliated College B (National University System) | High Burnout |

*Note on Qualitative Sampling Stratification and Tier Concordance*: The qualitative subsample ($N = 20$) was recruited via nested purposive maximum variation sampling across the quantitative survey burnout strata (`burnout_score` tiers: Low $n=2$, Medium $n=8$, High $n=10$). In Table 8, the 'Self-Reported Burnout Severity Tier' reflects the verified categorical classification obtained during the qualitative interview protocol (Question Q8), which exhibited 100% concordance with each participant's original quantitative survey sampling tier, confirming the temporal stability of the sampling strata on the day of the interview rather than serving as an independent blind validation.

### 8.1 Translation Procedure and Cross-Cultural Semantic Validation

Qualitative interviews were conducted in Bangla (the mother tongue of all participants) to ensure linguistic authenticity, psychological comfort, and cultural nuance. The first author (a native Bangla speaker, fluent in both languages) transcribed the audio recordings word-for-word into Bangla. Subsequently, the transcripts and illustrative narrative excerpts were translated from Bengali into English by the lead author for thematic coding, cross-case synthesis, and international reporting. To ensure translation fidelity and prevent semantic drift, a randomly selected 25% sub-sample of transcripts (n = 5) were independently back-translated into Bengali (van Nes et al., 2010; Regmi et al., 2010) by a second bilingual educational researcher. Discrepancies related to idiomatic expressions (e.g., context-specific expressions of psychological distress such as “chinta” [deep anxiety/worry] or religious coping markers such as “Alhamdulillah”) were resolved by consensus to guarantee cross-cultural semantic equivalence and conceptual validity. Where multi-part responses across related questions of the 10-item interview protocol are synthesized into illustrative excerpts in Sections 8.3–8.6, bracketed ellipses (`[...]`) clearly demarcate connected segments across conversational turns while maintaining strict semantic and tonal fidelity to the original spoken Bengali.

### 8.2 Reflexivity, Coder Independence and Inter-Rater Reliability

The primary author was involved in several aspects of the project (survey administration, machine learning pipeline development, and qualitative interviewing/coding). Because the interviewer was aware of participants' sampling strata during the semi-structured interviews to ensure maximum variation probing across burnout tiers, explicit methodological controls were instituted to mitigate the risk of confirmation bias: 1. Temporal Separation of Analytical Strands: Qualitative open and axial coding of all 20 transcripts was completed before running the final SHAP model interpretability pipeline. This strict separation of time allowed for an independent qualitative theme extraction without any knowledge of the quantitative feature importance rankings. 2. Inter-rater agreement: A second independent qualitative researcher, blind to the quantitative feature rankings and model predictions, analysed a 25% sub-sample of anonymised interview transcripts (n = 5). Inter-rater reliability was high across major thematic categories (Cohen’s $\kappa = 0.82$, 95% CI [0.74, 0.90]; Landis & Koch, 1977 [111]), reflecting strong thematic stability across independent coders.

### 8.3 Theme 1: Academic Dread and Career Despair

The interviews revealed that the most common fear was academic failure and the second most common fear was career prospects after graduation, mirroring the SHAP finding that academic performance indicators and CGPA are the most influential predictors of burnout. Students reporting High Burnout viewed academic demands as insurmountable pressure, not as manageable milestones. Participant 1 (Bachelor of Science in CSE 2nd Year, Private University A, High Burnout) summarized this experience: "My mental stress is quite high right now. Honestly, because I can't finish my daily tasks on time, the academic pressure feels even heavier. [...] I have constant anxiety about my career—what I'll do in the future, where my life is heading; thinking about these things makes me restless." Participant 10 (Bachelor 3rd Year, Private University D, High Burnout) expressed feeling overwhelmed by cumulative demands: "Mental stress is extremely high! Even if I study all day, it feels like the syllabus is never ending. [...] I am depressed all the time. There's the academic pressure, plus the current situation of the country—I don't see a good future." Participant 17 (Bachelor 1st Year, Private University G, Medium Burnout) noted the strict institutional grading context: "You already know the study pressure at our private university! I have to make it a rule to study 4-5 hours every day because our grading system is very strict. [...] Yes, I am often quite depressed about life and my studies."

### 8.4 Theme 2: Institutional Identity and Low-Pressure Burnout

A fascinating sub-theme emerged that challenged the traditional JD-R assumption that high academic demands are prerequisite for burnout. Some National University students reported low day-to-day curriculum pressure, but experienced severe burnout driven by institutional stigma and career despair. Participant 11 (Bachelor 1st Year, Affiliated College A [National University System], High Burnout) explicitly contrasted academic workload with career despair: "Brother, as you know, studying at the National University means there isn't that much study pressure. [...] Mentally, I am not doing very well. For one, I study at the National University, and on top of that, my CGPA is bad. I can't find any direction for my future, so my depression is quite high." Participant 20 (Bachelor 3rd Year, Affiliated College B [National University System], High Burnout) echoed this experience: "There's no direct pressure from studies, but since I don't study—there's a constant internal stress that works inside my mind. [...] Brother, I am depressed all day long. For one, I study at the National University, and on top of that, there's no clue about my career."

Methodological Caveat on Subgroup Evidentiary Base: It should be explicitly noted that this "low-pressure institutional burnout" theme emerged specifically from the two National University participants ($n = 2$, Participant 11 and Participant 20) represented in the purposive qualitative interview cohort ($N = 20$). While these accounts provide deep exploratory qualitative evidence of an unmeasured dimension of institutional marginalisation, this finding should be interpreted as exploratory and hypothesis-generating, warranting confirmatory investigation in dedicated future studies with larger cohorts of National University students.

### 8.5 Theme 3: Digital Fatigue as Maladaptive Coping

Burnout was found to be significantly associated with social media consumption (ANOVA $p < .001$). In qualitative interviews, we found that social media is frequently used as an avoidance mechanism for academic overwhelm, which subsequently disrupts time management and exacerbates the original stress, creating a vicious cycle of digital fatigue. Participant 18 (Bachelor 2nd Year, Private University H, High Burnout) described this habitual loss of digital control: "Social media is eating up all my valuable time. Between scrolling and web series, I waste at least 6 hours a day on my phone. [...] Burnout is very high. Between this daily routine and my mental state, I am completely tired." This unstructured digital consumption directly substituted for study and recuperative rest. Participant 8 (Bachelor 2nd Year, Private University A, High Burnout) similarly stated: "My whole day goes to my phone! Scrolling Facebook and playing PUBG takes up more than 6-7 hours a day. [...] I'm depressed quite often. No studies, no outdoor activities—adding it all up makes me feel like there is actually no future in Bangladesh."

### 8.6 Theme 4: Circadian Collapse and Sleep Deprivation (Biological Exhaustion)

Students experiencing severe burnout consistently reported their circadian rhythms breaking down completely from late-night digital consumption, academic anxiety, or employment obligations. This qualitative finding provides contextual support to the SHAP values identifying sleep hours and sleep quality as essential buffers. Participant 6 (Bachelor 2nd Year, Private University A, High Burnout) described the disruption of sleep hygiene: "Actually, I watch a lot of web series at night, so I go to bed late. Then I have to wake up in the morning for university, so my sleep schedule is completely off. [...] Very poor. I don't get enough time to sleep, and even when I get into bed, sleep doesn't come easily." Participant 12 (Bachelor 2nd Year, Private University A, High Burnout) emphasized the compounded strain of employment and disrupted rest: "The academic pressure feels like a lot to me. I absolutely cannot balance the job and my studies together. [...] My sleep situation is very pathetic. I only get the chance to sleep for about 5-6 hours a day. [...] I am a very light sleeper, and it breaks at the slightest sound." In stark contrast, students who maintained consistent sleep hygiene and physical activity exhibited resilience even under high academic workloads. Participant 5 (Master's 1st Year, Private University C, Low Burnout) managed academic coursework, research, and graduate assistantship responsibilities with consistent restorative sleep: "Alhamdulillah, I can sleep very peacefully. I make it a rule to sleep 7-8 hours every day. [...] My burnout level is quite low. I am managing everything beautifully, and my results are going well too." Participant 4 (Bachelor 3rd Year, Private University A, Medium Burnout) highlighted the protective buffer of physical exercise and deep sleep: "Yes, I exercise regularly every morning. I also play football. [...] Alhamdulillah, my sleep is very good. I get deep sleep."

### 8.7 Qualitative Results Summary

We humanise machine learning data via expanded thematic analysis. It reveals that for this cohort burnout is an expression of an integrated risk pathway: high academic and career anxiety is associated with psychological distress which students attempt to cope with by engaging in digital media (often 5-7 hours per day). This digital use replaces their sleep architecture and they are physically and mentally unfit to face the daily academic demands. Moreover, it points out the special circumstances of students in the National University system, who are worn out not by a heavy academic burden but by the stigma of attending a National University.

## 9. Mixed Methods Integration and Extended Discussion

The main advantage of an explanatory sequential mixed-methods design (QUAN → QUAL) is the triangulation, or systematic combination of different data streams to develop a cohesive theoretical model [26]. For this study, the machine learning algorithms (i.e., the SHAP feature importances extracted from the Random Forest model in Section 7) were validated against the deeply contextual thematic analysis of the 20 qualitative interviews (Section 8) for their computational objectivity. The triangulation process shows significant overlap between the algorithmic results and the psychological experiences reported by students. Taken together, these findings provide important integrated insights into the functioning of academic burnout. Table 9 presents the formal explanatory triangulation matrix mapping quantitative SHAP predictors to qualitative themes.

### Table 9. Explanatory Triangulation Matrix: SHAP Quantitative Predictors Mapped to Qualitative Themes

| SHAP Feature / Quantitative Signal | Mean \|SHAP\| Rank | Emergent Qualitative Theme | Representative Participant Evidence | Triangulation Status |
|---|---|---|---|---|
| Academic Performance Index | #1 (0.0388) | Theme 1: Academic Dread and Career Despair | P1: "I have constant anxiety about my career — what I'll do in the future"; P10: "Even if I study all day, it feels like the syllabus is never ending" | **Convergence**: Algorithmic signal corroborated by all High Burnout participants |
| CGPA Midpoint | #2 (0.0362) | Theme 1 & Theme 2: Career Despair / Institutional Identity Burnout | P11: "I study at the National University, and my CGPA is bad — I can't find any direction" | **Convergence + Extension**: Qualitative adds institutional stigma dimension unmeasured by CGPA alone |
| Screen-to-Sleep Ratio | #3 (0.0298) | Theme 3: Digital Fatigue as Maladaptive Coping | P18: "Social media is eating up all my valuable time — at least 6 hours a day"; P8: "My whole day goes to my phone" | **Convergence**: Sequential digital-sleep displacement mechanism corroborated qualitatively |
| Burnout Vulnerability Index | #4 (0.0238) | Theme 1 & Theme 4: Psychological Distress + Sleep Collapse | P6: "My sleep schedule is completely off"; P12: "My sleep situation is very pathetic" | **Convergence**: Demand-resource imbalance operationalized in both analytical strands |
| Social Media Hours | #5 (0.0202) | Theme 3: Digital Fatigue as Maladaptive Coping | P8: "Scrolling Facebook and playing PUBG takes up more than 6-7 hours a day" | **Convergence**: Consistent across SHAP and all High Burnout interviews |
| Sleep Deprivation Index / Sleep Quality | #7–8 | Theme 4: Circadian Collapse and Biological Exhaustion | P5 (Low Burnout): "I make it a rule to sleep 7-8 hours every day — my burnout level is quite low" | **Complementarity**: Protective role of sleep hygiene identified qualitatively as buffer |
| Gender / Age / Degree Level | #Bottom (\<0.007) | Themes 1–4 (Universal): All demographic groups report behavioural, not demographic, burnout drivers | No participant attributed burnout to age or gender | **Convergence**: Demographic invariance confirmed across both strands |
| Institutional type (unmeasured quantitatively) | Not captured | Theme 2: Institutional Identity and Low-Pressure Burnout | P11, P20: National University stigma as burnout driver independent of academic workload | **Divergence / Extension**: Qualitative reveals mechanism invisible to the quantitative model |

Note. Triangulation status definitions: *Convergence* = quantitative and qualitative findings directly corroborate; *Complementarity* = qualitative adds depth or nuance to quantitative signals without contradiction; *Divergence/Extension* = qualitative identifies mechanisms absent from the quantitative feature space, representing unmeasured constructs for future model iteration. Protocol adapted from Farmer et al. (2006) [112].

### 9.1 CGPA Convergence and Career Disenchantment

Quantitative Signal: Global SHAP attribution identified `academic_performance_index` (Mean |SHAP| = 0.0388), `cgpa_midpoint` (Mean |SHAP| = 0.0362), `screen_to_sleep_ratio` (Mean |SHAP| = 0.0298), and `burnout_vulnerability_index` (Mean |SHAP| = 0.0238) as the primary mathematical predictors of High Burnout classification. Qualitative Context: Reflexive thematic analysis contextualises the algorithm's prioritisation of these academic metrics. For university students in Bangladesh, CGPA operates as a high-stakes proxy for post-graduate career viability in an intensely competitive employment market. Participants consistently articulated a bidirectional psychological burden: striving to sustain a competitive CGPA incurs severe cognitive and emotional exhaustion, whereas sub-optimal academic standing engenders acute anxiety regarding post-graduate unemployment. Furthermore, qualitative inquiry illuminated an institutional nuance absent from the quantitative feature space: students enrolled in the National University system reported severe burnout even under nominal curricular workloads, driven by perceived institutional marginalisation and credential devaluation. This indicates that perceived academic pressure is inextricably intertwined with anticipatory career despair.

### 9.2 Convergence of Screen-to-Sleep Disruption and Digital Fatigue Dynamics

Quantitative Signal: SHAP analysis identified screen_to_sleep_ratio (Mean |SHAP| = 0.0298) as the third most important burnout predictor and social media hours (Mean |SHAP| = 0.0202) as the fifth, suggesting that digital displacement of biological sleep recovery is a separate and statistically potent burnout mechanism — independent of academic performance indicators. Qualitative Context: This computational relationship is contextualised by the qualitative interviews, revealing a sequential maladaptive coping pathway. Students do not passively consume social media. Rather, with increasing academic stress and psychological distress, social media use (often 5-7 hours a day) is an escapist coping strategy for academic overwhelm. This digital engagement directly displaces sleep duration, compounding the original stress with chronic biological exhaustion. "Actually, I watch a lot of web series at night, so I go to bed late. Then I have to wake up in the morning for university, so my sleep schedule is completely off," explained Participant 6 (Bachelor 2nd Year, Private University A, High Burnout). Participant 8 echoed the same compounding dynamic: "My whole day goes to my phone! Scrolling Facebook and playing PUBG takes up more than 6-7 hours a day. [...] No studies, no outdoor activities—adding it all up makes me feel like there is actually no future in Bangladesh." The Random Forest model captured this compounding interaction mathematically: the combination of high social media use and low sleep hours is highly correlated with the High Burnout classification. Crucially, the qualitative strand shows that this pattern is a product of psychological avoidance behaviour, not mere lifestyle preference – a mechanistic subtlety that the quantitative model picks up as a statistical signal but cannot encode as an intervention target without qualitative contextualisation.

### 9.3 Sociodemographic Invariance in Algorithmic and Qualitative Burnout Risk Attribution

Quantitative Signal: Demographic variables (Gender, Age Group, Degree Level) were found at the bottom of SHAP hierarchy and had mathematically insignificant predictive power. Qualitative Context: The interviews corroborate this observed invariance. Students did not attribute exhaustion to age or gender in their accounts. Instead, burnout was universally attributed to behavioural demands (working part-time jobs, studying long hours) and psychological states (depression over career prospects). The model's negligible attribution of burnout risk to age and gender is consistent with the qualitative finding that students attributed exhaustion to behavioural and psychological demands rather than fixed demographic characteristics — a pattern that corroborates the algorithmic output through independent narrative evidence.

### 9.4 Toward an Integrated Theoretical Model of Student Burnout

The study triangulates the SHAP values with qualitative themes and proposes a contextualised, data-driven adaptation of the Job Demands-Resources (JD-R) framework for student burnout in the South Asian higher education context. The integrated evidence is consistent with a hypothesised compounding risk pathway along the JD-R health impairment trajectory: Systemic Academic/Career Anxiety (CGPA pressure or institutional stigma) may contribute to primary Psychological Distress (depression and stress). Students unable to engage adaptive strategies may subsequently exhibit Maladaptive Digital Escapism (high social media use of 5–7 hours daily), which may further co-occur with Biological Exhaustion (sleep deprivation to 5–6 hours). This triangulated adaptation suggests that machine learning algorithms, suitably deconstructed through Explainable AI and contextualised through human interviews, can contribute meaningfully to our understanding of complex psychological phenomena. This proposed sequential pathway is inherently cross-sectional and correlational in the current study; the temporal ordering and directionality of these relationships require confirmation in future longitudinal studies using structural equation modelling on independent samples.

### 9.5 Research Hypothesis & Research Questions Testing

We now explicitly and empirically close the conceptual and predictive framework set in Section 1 by formally testing the four research hypotheses (H1–H4) and four research questions (RQ1–RQ4). Testing Research Hypotheses: H1 (Algorithm Superiority): **Partially Supported** — Bagging ensembles (Random Forest) significantly outperformed the single-learner baseline (Decision Tree); gradient boosting ensembles did not. H1 predicted that ensemble-based classifiers (Random Forest, XGBoost, LightGBM, CatBoost) would perform better than single-learner models. The hypothesis is partially supported: the bagging-based ensemble (Random Forest) significantly outperformed the single learner (Decision Tree) via McNemar test, confirming one class of ensemble superiority. However, gradient boosting ensembles (XGBoost, LightGBM, CatBoost) did not significantly outperform Logistic Regression or Random Forest — an important boundary condition of the hypothesis. XGBoost achieved Accuracy = 62.40%, F1 = 0.5292, which is less than that of Logistic Regression (Accuracy=64.39%, F1=0.5348). LightGBM scored 63.39% accuracy, again underperforming the logistic regression baseline. CatBoost (65.06%, F1 = 0.5455) did slightly better but was not statistically significantly better than Logistic Regression (McNemar p > 0.45). This outcome is methodologically informative, and is driven by two synergistic factors: (1) Sample Size and Algorithm Complexity: At N = 601, the variance reduction of randomised bagging (Random Forest, which averages over uncorrelated decorrelated trees) reliably outperforms sequential boosting, which typically requires N ≥ 2,000–5,000 instances to exploit weak-learner gradient stacking without fitting subjective self-report noise [89]; (2) Shrinkage Dynamics on Small Tabular Data: XGBoost's standard default learning rate ($\eta = 0.30$) with tree depth 6 is relatively aggressive gradient descent updates, predisposing sequential decision stumps to localised overfitting on self-report psychometric items compared to shallow regularised logistic regression. Although a fine-grained hyperparameter grid search (e.g., finer shrinkage $\eta \in [0.01, 0.05]$ with substantial $\text{L}_1/\text{L}_2$ regularisation) can reclaim marginal gains, the boosting models in the main multi-model benchmark maintained fixed default parameters consistent with the original protocol. Subsequent nested CV tuning corroborated performance stability for the champion Random Forest and Logistic Regression (Section 7.1), but a systematic assessment of deep hyperparameter grids for all gradient boosting architectures is a topic for future work. Hence, the hypothesis is only partially supported, as Random Forest (bagging ensemble, Accuracy = 65.89%) and the Soft Voting Ensemble (65.89%) performed significantly better than the single-learner Decision Tree in unadjusted McNemar tests ($\chi^2 = 4.73, p = 0.0297$ and $\chi^2 = 4.66, p = 0.0308$ respectively). This explicit falsification of H1 for boosting models emphasises the importance of selecting algorithms that are suitable for the sample size in educational data mining. H2 (Psychological Demand Ranking): Supported. H2 predicted that psychological demand features would rank in the top five predictors. The SHAP analysis indicated that the Academic Performance Index (#1), CGPA Midpoint (#2), Screen-to-Sleep Ratio (#3), Burnout Vulnerability Index (#4), and Social Media Hours (#5) were in the top spots. H3 (Resource Depletion Pathway): Partially Supported. H3 postulated that resource-related features (sleep quality score, physical activity hours, wellbeing buffer index) would be significantly and negatively associated with burnout severity. In univariate statistical testing (Section 5), the resource indicators such as motivation score showed a non-significant trend (p = 0.084). However, in non-linear multi-variate modelling, SHAP feature dependence plots (Section 7) confirmed that biological recovery deficits (sleep hours and sleep quality) and composite well-being buffer depletion strongly accelerated burnout risk when interacting with high stress. It suggests that the issue of resource depletion is not a matter of simple bivariate main effects but rather complex non-linear feature interactions [38]. H4 (Triangulated Convergent Validity): Supported. Qualitative themes would corroborate the quantitative feature hierarchy. Triangulation analysis (Sections 9.1-9.4) validated strong qualitative alignment. Qualitative narratives of CGPA dread, digital bingeing, and sleep loss directly reflected top SHAP mathematical attributions. Answering the research questions RQ1 (Best Predictive ML Model): This question is answered in Section 6. Random Forest got maximum Cross-Validated Accuracy (65.89%) and maximum ROC-AUC (0.7126), Soft Voting Ensemble got same Accuracy (65.89%), ROC-AUC (0.7069) and maximum F1-score (0.5514) and LightGBM got maximum Recall (0.5059) over 10-Fold Stratified Cross-Validation on N = 601. RQ2 (Most Influential Predictors & XAI Convergence) Section 7. Most important global factors were Academic Performance Index (Mean |SHAP| = 0.0388) and CGPA midpoint (0.0362) followed by Screen-to-Sleep Ratio (0.0298) and Burnout Vulnerability Index (0.0238) while demographic variables contributed negligibly (< 0.007). RQ3 (Qualitative Experiential Themes): Section 8 presents four major themes that emerged: (1) Academic dread and career despair, (2) Institutional identity and “low-pressure” burnout, (3) Digital fatigue as maladaptive coping, and (4) Circadian collapse and sleep deprivation (biological exhaustion). RQ4 (Triangulated Meta-Inferences & Policy Implications): Discussed in Sections 9 & 10. Triangulation produced a unified 4-stage compounding burnout model (Academic/Career Anxiety $\rightarrow$ Psychological Distress $\rightarrow$ Maladaptive Digital Escapism $\rightarrow$ Circadian Collapse) to inform algorithm-guided institutional interventions.

### 9.6 Theoretical and Clinical Significance

The main purpose of this research was to develop an interpretable exploratory predictive model of university student burnout by integrating supervised machine learning and reflexive qualitative analysis. This research goes beyond traditional isolated statistical approaches and employed Explainable AI (SHAP) to unpack the algorithmic logic and triangulated these mathematical findings with human lived experiences. The findings offer useful insights into the patterns of academic exhaustion, particularly in the high-stakes educational context of Bangladesh.

### 9.7 Predictive Modelling & Algorithm Selection

The machine learning pipeline showed that burnout is a predictable psychometric outcome based on behavioural and psychological survey features. The best overall accuracy was achieved by the Random Forest classifier (65.89%, ROC-AUC = 0.7126). The Soft Voting Ensemble provided the best predictive performance (65.89% Accuracy, ROC-AUC = 0.7069, F1 = 0.5514) at the same level. For high-risk burnout detection, LightGBM achieved the highest recall (0.5059) considering single classifiers with default thresholding. These measures are relevant for educational psychology survey prediction. Primary self-reported psychometric data is characterised by inherent subjective human variability and behaviour noise, unlike clinical or laboratory settings with rigid biomedical sensors [100]. A 10-fold cross-validated accuracy of 65.89% with a ROC-AUC of 0.7126 thus marks a modest, leak-free real-world exploratory screening baseline (8.32 percentage points above the 57.57% majority baseline). The ROC-AUC value of 0.71 represents an acceptable and statistically significant predictive signal, supporting the existence of non-linear identifiable patterns between academic burnout and the behavioural feature space. Model Parsimony, Occam's Razor and Baseline Equivalence: Pairwise McNemar hypothesis testing (Section 6.3) showed that the difference in performance between Random Forest (Accuracy = 65.89%, F1 = 0.5199) and standard Logistic Regression (Accuracy = 64.39%, F1 = 0.5348) was not statistically significant ($\chi^2 = 0.51, p = 0.4743$). Complex gradient-boosted models such as CatBoost (65.06%) also did not significantly outperform Logistic Regression ($\chi^2 = 0.07, p = 0.7925$) or Random Forest ($\chi^2 = 0.23, p = 0.6350$). This finding reflects two important practical considerations when evaluated against the principle of parsimony (Occam's razor): 1. Pragmatic Deployment Baseline: Regularised Logistic Regression is a very efficient, statistically comparable baseline algorithm, for resource-constrained university registrar systems, or mobile wellness applications, that require real-time scoring without non-linear ensemble dependencies. 2. Why Non-Linear Ensembles & SHAP. Even when globally equally accurate, tree-based ensembles (Random Forest) are still needed for two key analytical capabilities: (a) capturing non-monotonic interaction split-points (e.g. non-linear risk escalation above certain screen-to-sleep ratios) without requiring manual polynomial specification in a linear model, and (b) enabling scale-invariant non-parametric SHAP feature attributions that are mathematically robust under correlated composite features (Section 5.4). Benchmarking several classifiers provides an empirical validation step showing non-linear models can generalise competitively without overfitting on noisy self-report data.

### 9.8 Deconstructing Burnout in the JD-R Model

The empirical findings provide theoretical corroboration and contextual extension of the Job Demands-Resources (JD-R) model [38] to tertiary education in a developing economy. The JD-R model posits that burnout emerges when structural demands chronically outstrip restorative resources. Triangulating game-theoretic SHAP feature attributions with qualitative narratives identified three prominent demand-resource vectors: 1. Academic Performance (CGPA) as Primary Demand Vector: Unlike Western higher education cohorts where external employment or social isolation frequently dominate burnout risk [14], our XAI model identified Academic Performance Index (Mean |SHAP| = 0.0388) and CGPA Midpoint (Mean |SHAP| = 0.0362) as the foremost risk attributions. Qualitative narratives contextualised this ranking: in Bangladesh's constrained graduate job market, academic standing is perceived as the sole determinant of economic security. The substantial psychological strain associated with maintaining a high CGPA, combined with the hopelessness associated with sub-optimal grades (especially among stigmatised National University students), constitutes an unremitting structural demand. 2. The Digital Fatigue Paradox: While recreational activity theoretically functions as a restorative psychological resource, unstructured social media consumption (>5 hours daily) emerged empirically as a demand-amplifying factor (Mean |SHAP| = 0.0202; Screen-to-Sleep Ratio Mean |SHAP| = 0.0298). Qualitative analysis revealed that digital engagement operates as an avoidance-based coping strategy that displaces sleep rather than replenishing cognitive resources. 3. Biological Resource Depletion (Sleep Deprivation): Objective and self-reported sleep metrics demonstrated prominent negative associations with burnout severity. Triangulation identified a consistent behavioural co-occurrence pattern wherein academic anxiety and digital escapism concurrently restrict sleep duration to 5–6 hours, inducing a chronic biological resource deficit that undermines next-day academic coping capacity.

### 9.9 Contributions of Methodology

This research makes an important methodological contribution, because educational psychology has always used either quantitative linear statistics that do not take into account complex interactions between variables or qualitative interviews that are not scalable for prediction. This study presents a promising analytical approach in an explanatory sequential mixed-method design (QUAN → QUAL) combining Machine Learning, Explainable AI (SHAP), and Thematic Analysis. Results demonstrate that algorithms can detect meaningful predictive signals about psychological states, XAI is able to extract the theoretical hierarchy supporting those predictions, and qualitative research can provide the essential human context to explain why such hierarchy exists. The triangulated framework enables the interpretation of algorithmic outputs, making advanced computational frameworks actionable and interpretable for mental health professionals and university administrators.

### 9.10 Limitations and Future Research Directions

#### 9.10.1 Limitations of the Method

1. Cross-Sectional Design: The quantitative data (N = 601) were collected in a cross-sectional design. Although the machine learning algorithms found strong predictive patterns and qualitative interviews provided experiential insights into self-perpetuating behavioural cycles (e.g., social media perceived as displacing sleep), a cross-sectional design is limited in its ability to determine causality. For example, it is still possible in theory that severe burnout leads to more social media use rather than vice versa. 
2. Self-Report Bias: The quantitative survey and qualitative interviews were entirely based on self-reported metrics. Variables such as `study_hours_numeric` and `sleep_hours_numeric` remain subject to recall bias. 
3. Unmeasured Macro-Environmental Factors: The machine learning algorithms were trained on specific academic and behavioural metrics. However, these naturally leave out the broader socio-economic and environmental realities specific to Bangladesh. The survey did not capture factors such as extreme traffic congestion (which can drain energy from daily life), sudden political instability, university closures, natural disasters, and recent economic inflation [11, 123]. Such external “natural” stressors probably play a massive, hidden role in the high burnout rates, especially in Dhaka, which the current algorithm cannot account for. 
4. Modest statistical predictive signal and threshold trade-offs: The best random forest achieved an accuracy of 65.89% (~8.3 percentage points above the 57.57% majority baseline) and ROC-AUC of 0.7126 with a default threshold recall of 43.53%. While statistically significant, these results indicate that cross-sectional self-report survey data offer a modest exploratory screening signal rather than a diagnostic decision boundary. With higher sensitivity ($th=0.38$, Recall = 71.76%, Table 6), more cases are identified, but the specificity is lower (56.07%). This emphasises that algorithmic risk stratification should serve solely as a non-binding decision support indicator, augmented by human clinical judgement. 
5. Engineered composite features and Construct Adjacency: The 9 composite features operationalize theoretical JD-R demands and COR resources (`academic_performance_index` $r = -0.246$, `cgpa_midpoint` $r = -0.264$, `screen_to_sleep_ratio` $r = 0.219$, `burnout_vulnerability_index` $r = 0.183$, `motivation_deficit_score` $r = 0.151$, `wellbeing_buffer` $r = -0.076$). While feature ablation (Section 6.1.1) confirmed that composite indices provide genuine empirical gains for non-linear tree models (+1.84% accuracy and +0.0238 AUC for Random Forest) without direct target leakage, these indices are construct-adjacent to the target, measuring concurrent behavioural and psychological strain rather than orthogonal physiological biomarkers. 
6. Authors’ Reflexivity and Dual-Role Limitation: The first author was responsible for survey collection, ML pipeline development, and qualitative coding. We used temporal separation (open qualitative coding prior to SHAP execution) and a 25% independent inter-rater check (Cohen’s $\kappa = 0.82$). Because qualitative sampling was stratified across pre-assigned survey burnout tiers, 100% concordance with interview-day self-reports reflects stratum stability rather than an independent blind validation.
7. Sample Size and Need for External Multi-Center Validation: The sample size (N = 601) is modest for training complex ML architectures. Nonetheless, 10-fold stratified cross-validation provided robust out-of-fold generalisation estimates and cross-subgroup pseudo-external validation supported stability across academic degree levels (Section 6.5, Table 7, Bachelor’s vs. Master’s/PhD/Diploma cohorts). However, true geographically-independent external validation on multi-center cohorts remains essential. Future multi-institutional studies should evaluate external generalisability across independent geographical cohorts in South Asia and beyond, consistent with TRIPOD guidelines [124, 125]. Moreover, the cohort was restricted to Bangladeshi universities, and thus the specific institutional dynamics – including the stigma of the National University system – may not directly translate to Western higher education contexts. 
8. Qualitative Subgroup Sample Constraints: Similarly, the emergent qualitative theme of 'low-pressure institutional burnout' was identified from the two National University participants ($n = 2$) included in the purposive interview sample ($N = 20$). While providing rich hypothesis-generating context on institutional stigma, this sub-group finding should be interpreted as exploratory pending larger dedicated investigations.
9. Multi-Institutional Clustering and Non-Independence of Observations: Survey participants were recruited across 11 distinct higher education institutions (8 private, 1 public, 2 National University affiliated colleges). Standard classification algorithms and inferential statistics treated the sample as pooled independent observations without explicit hierarchical linear modeling or institutional random effects. Unmodeled intraclass correlations (ICC) across campuses represent an unmeasured source of institutional clustering that multi-level modeling should address in future work.
10. Unmeasured Disciplinary Confounders: Academic major (e.g., Computer Science and Engineering vs. Humanities vs. Business) was omitted from the quantitative survey to safeguard student anonymity, despite qualitative evidence that engineering students experience distinct curriculum and grading pressures. Future survey administrations should include broad discipline categorizations.
11. Discrete Interval-Midpoint Scaling: Daily time budgets (study, sleep, social media, exercise), attendance, and CGPA were collected as categorical interval brackets to maximize response completion and subsequently converted to discrete midpoints (Table 2). While effective for exploratory survey modeling, these binned approximations lack the granularity of continuous sensor-derived time tracking.
12. Qualitative Low Burnout Subsample Under-Representation: The purposive qualitative sample (N = 20) was stratified to prioritise participants with moderate and high burnout severity, resulting in only n = 2 Low Burnout participants (P5, P7). While this distribution reflects the quantitative prevalence imbalance in the original sample (Low Burnout: 19.47%), it substantially limits the interpretive evidence base for protective-factor analysis and comparisons between low- and high-burnout experience. Any qualitative claims regarding "protective" lifestyle patterns in Low Burnout students are derived from a thin evidential base and should be interpreted with corresponding caution.

#### 9.10.2 Directions for Future Research

1. Longitudinal and sensor-based data: Moving from cross-sectional survey to longitudinal monitoring should be a focus for future research. Wearable fitness trackers to capture objective sleep architecture and screen-time monitoring applications to capture precise digital consumption would eliminate self-report bias and provide higher-fidelity inputs for deep learning models. 2. Multi-Center External Validation and Cross-Cultural Portability: The stability of the model across academic degree tiers was validated using 10-fold cross-validation and cross-subgroup pseudo-external validation (Section 6.5). However, the cross-cultural portability of the engineered feature indices needs to be validated across independent multi-center university cohorts in South Asia and globally in future studies. 3. Intervention Studies: Educational data mining is primarily aimed at informing interventions. Future work should build on the identified SHAP hierarchy (targeting CGPA anxiety and sleep hygiene) to inform specific, algorithm-guided psychotherapeutic interventions at university counselling centers and to evaluate the efficacy of these interventions in reducing systemic burnout through randomised control trials.

## 10. Conclusion and Policy Recommendations

Academic burnout among university students is rapidly becoming a systemic public health crisis, but conventional diagnostic frameworks are still limited by linear statistics and isolated methodological silos. This study aimed to overcome these limitations by using an explanatory sequential mixed-methods design (QUAN → QUAL) that combined quantitative ML-based prediction with systematic qualitative inquiry. 
This research trained ten distinct supervised machine learning algorithms on raw psychometric survey data ($N=601$) and identified a modest, statistically significant, predictive signal for student burnout. The best-performing Random Forest and Soft Voting Ensemble models achieved cross-validated accuracy of 65.89% (ROC-AUC = 0.7126) and 65.89% (ROC-AUC = 0.7069) respectively on inherently noisy self-report data — representing an 8.32 percentage-point gain over the majority baseline and a meaningful exploratory screening signal above chance-level prediction. 
But prediction does not suffice for intervention. Explainable AI (SHAP) was used to interpret algorithmic outputs, resulting in a feature importance hierarchy of burnout-related variables. SHAP analysis showed demographic characteristics (age or gender) played a minor role in prediction. Instead, the analysis found the mathematical epicentre of academic exhaustion was CGPA pressure, depression, unstructured social media use, and sleep deprivation. 
Subsequent thematic analysis of 20 in-depth qualitative interviews contextualised this computational hierarchy. The qualitative stream showed strong convergence with the SHAP values and identified a consistent co-occurrence pattern: students reporting extreme career anxiety (CGPA pressure or institutional stigma) concurrently reported high social media use (often >5 hours/day) as a maladaptive coping mechanism, which appeared to displace restorative sleep and co-occur with chronic burnout. Crucially, the qualitative analysis also identified an exploratory finding of “low-pressure burnout” among National University students, in which exhaustion appeared to be associated with institutional marginalisation and career hopelessness rather than heavy academic workload — a hypothesis-generating pattern warranting confirmatory investigation in dedicated future studies. 
In sum, the present study offers a contextualised and data-driven adaptation of the JD-R model to the extant burnout literature. This suggests algorithms can discover significant predictive patterns, but human context is important for intervention design. Given the growing severity of mental health crises on campuses, we suggest the following evidence-based policy recommendations.

### 10.1 Policy Recommendations

Based on the integrated quantitative-qualitative findings, the following specific, actionable policy recommendations are offered for Bangladeshi higher education institutions:

1. Implement Sleep Hygiene and Digital Wellness Programs. SHAP analysis identified screen-to-sleep ratio (Mean |SHAP| = 0.0298) and social media hours (|SHAP| = 0.0202) among the top 5 burnout predictors. Universities should integrate structured digital wellness workshops — covering sleep hygiene protocols, device-use curfews (no screens after 10 PM), and screen-time monitoring applications — into first-year orientation programs and repeat them at each academic year transition.

2. Deploy CGPA-Triggered Voluntary Counseling & Mentoring Protocols. The academic performance index (|SHAP| = 0.0388) and CGPA midpoint (|SHAP| = 0.0362) are the strongest quantitative burnout indicators. University registrar systems should automatically flag students whose semester GPA falls below 2.5 for a voluntary initial counseling or academic mentoring consultation — strictly framed as supportive academic guidance rather than mandatory mental health referral, thereby avoiding stigma-driven avoidance while respecting model false-positive bounds (~45% false positive rate at high sensitivity thresholds).

3. Develop Dedicated Support Programs for National University Students. The qualitative analysis revealed a distinct institutional identity burnout pathway among National University students driven by institutional stigma and career hopelessness rather than study overload. Dedicated peer-support networks, career orientation workshops, and alumni mentoring programs specifically designed for this population represent a high-priority institutional gap requiring targeted investment.

4. Calibrate Algorithmic Early-Warning Thresholds for Voluntary Triage. The threshold-tuning analysis demonstrates that at th = 0.38 (Sensitivity = 71.76%), 183 of 255 high-risk students are identified at the cost of elevated false-positive referrals. Universities with sufficient counseling staff should deploy the high-recall threshold, while resource-constrained institutions may prefer the default threshold (th = 0.50, Specificity = 82.37%). Any algorithmic deployment must explicitly communicate to counselors and students that algorithmic flags constitute a preliminary, non-binding decision-support indicator, not a clinical diagnosis or automated mandatory action trigger.

5. Pilot an Ethics-Compliant Algorithmic Early-Warning Dashboard. Future collaboration between university IT departments and student mental health services should pilot an anonymized burnout risk dashboard — populated quarterly from academic records (CGPA, attendance), library utilization data, and opt-in wellness surveys — at 2–3 Bangladeshi institutions. Pilot evaluation should assess both predictive accuracy under real-world conditions and student and counselor acceptance of algorithmic triage, with results informing national higher education mental health policy.
 
6. Translate Empirical Models into Accessible Web-Based Screening Tools. To bridge the gap between academic machine learning research and immediate student support, the study's champion Random Forest classifier, composite index formulations, calibrated screening thresholds ($\theta = 0.38$), and game-theoretic SHAP feature-attribution engine have been implemented as an open-access, privacy-first web application ("Burnout Radar", publicly hosted at: `https://burnoutwebapplication.netlify.app/`). This platform allows university students to privately evaluate their behavioral risk factors (such as screen-to-sleep displacement) without transmitting or storing personal data, providing personalized mitigation strategies alongside verified Bangladesh mental health crisis contacts (e.g., Kaan Pete Roi: +880 1779 554391, National Mental Health Helpline: 16122).

## Declarations

### Ethics Approval and Consent to Participate

This study was conducted in accordance with the ethical principles outlined in the Declaration of Helsinki for research involving human subjects. The target population comprised university students enrolled in tertiary education institutions in Bangladesh, including a freshman transition cohort (aged 17–18 years) who had completed Higher Secondary Certificate (HSC) examinations and transitioned into undergraduate study. Because the research involved an observational, non-invasive cross-sectional survey and voluntary qualitative interviews carrying minimal psychological risk, participation was strictly voluntary. Formal ethics committee review was exempt under institutional minimal-risk educational research guidelines of the Department of Computer Science and Engineering, Presidency University, Dhaka-1212, Bangladesh (exemption category: anonymous observational survey research involving no deception, no sensitive personal data collection beyond academic self-report, and no intervention). Electronic informed consent was explicitly obtained from all respondents prior to survey participation and qualitative interviewing, explaining the study purpose, voluntary nature, right to withdraw, and data handling procedures. Full anonymity was strictly maintained across all published data and quotes using standardized pseudonyms, and de-identified data were stored securely in encrypted storage.

### Consent for Publication

All participants granted explicit consent for anonymized statistical data and verbatim qualitative quotes to be published in peer-reviewed scientific journals.

### Availability of Data and Materials

The complete de-identified quantitative survey dataset (`Quantitative_Survey_Data.xlsx`), anonymized qualitative interview codebook and transcript excerpt table (`Qualitative_Interview_Transcripts_Anonymized.pdf`), modular feature engineering script (`feature_engineering.py`), machine learning training script (`train_ml.py`), SHAP evaluation pipeline (`run_shap.py`), exact runtime dependency manifest (`requirements.txt`), manuscript source (`Manuscript_Student_Burnout.md`), and research repository are publicly deposited and freely available in an open-access GitHub repository at: `https://github.com/rifatmiah92/student-burnout-ml-mixed-methods`. Furthermore, the interactive, client-side web screening application is deployed and publicly accessible at: `https://burnoutwebapplication.netlify.app/` (with source code repository at: `https://github.com/rifatmiah92/Burnout-web-application`).

### Competing Interests / Conflict of Interest Statement

The authors declare that they have no competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### Funding Statement

This research received no specific grant or financial support from any funding agency in the public, commercial, or not-for-profit sectors. All research activities were conducted independently by the authors as part of academic scholarship activities. The study was funded entirely by the authors' own resources. *(TRIPOD Item 18 Compliance: No external funding was received for this study.)*

### Acknowledgements

The authors wish to thank all 601 survey participants and the 20 qualitative interview participants for generously contributing their time and experiences to this research. We thank the 11 participating higher education institutions in Bangladesh for facilitating student access. We thank the anonymous second qualitative coder for their independent coding contribution (inter-rater reliability κ = 0.82). We acknowledge that the open-source scientific ecosystem — particularly scikit-learn, SHAP, Python, and the Netlify deployment platform — made the computational and deployment components of this research freely accessible.

### Authors' Contributions

Rifat Miah conceptualized the study, performed primary survey data collection and qualitative interviews, engineered features, implemented the machine learning models and SHAP evaluation pipelines, conducted statistical analyses, and drafted the manuscript. Dr. A.S.M. Shihavuddin supervised the research, refined the analytical methodology, provided technical and theoretical oversight in machine learning and computer vision/data science applications, critically reviewed and edited the manuscript, and approved the final submission. Both authors read and approved the final manuscript.

# References

[1] Madigan, D. J., & Curran, T. (2021). Does burnout affect academic achievement? A meta-analysis of over 100,000 students. Educational Psychology Review, 33(2), 387-405. https://doi.org/10.1007/s10648-020-09533-1

[2] Maslach, C., & Leiter, M. P. (2016). Understanding the burnout experience: Recent research and its implications for psychiatry. World Psychiatry, 15(2), 103-111. https://doi.org/10.1002/wps.20311

[3] Salmela-Aro, K., & Read, S. (2017). Study engagement and burnout profiles among Finnish higher education students. Burnout Research, 7, 21-28. https://doi.org/10.1016/j.burn.2017.11.001

[4] Vizoso, C., Arias-Gundin, O., & Rodriguez, C. (2019). Coping, academic engagement and performance in university students. Higher Education Research & Development, 38(7), 1515-1529. https://doi.org/10.1080/07294360.2018.1504006

[5] World Health Organization. (2019). International statistical classification of diseases and related health problems (11th ed.). World Health Organization. https://icd.who.int/

[6] Maslach, C., Schaufeli, W. B., & Leiter, M. P. (2001). Job burnout. Annual Review of Psychology, 52(1), 397-422. https://doi.org/10.1146/annurev.psych.52.1.397

[7] Schaufeli, W. B., Martinez, I. M., Pinto, A. M., Salanova, M., & Bakker, A. B. (2002). Burnout and engagement in university students: A cross-national study. Journal of Cross-Cultural Psychology, 33(5), 464-481. https://doi.org/10.1177/0022022102033005003

[8] Frajerman, A., Morvan, Y., Krebs, M. O., Gorwood, P., & Chaumette, B. (2019). Burnout in medical students before residency: A systematic review and meta-analysis. European Psychiatry, 55, 36-42. https://doi.org/10.1016/j.eurpsy.2018.08.006

[9] Erschens, R., Keifenheim, K. E., Herrmann-Werner, A., Loda, T., Schwille-Kiuntke, J., Bugaj, T. J., Nikendei, C., Huhn, D., Zipfel, S., & Junne, F. (2019). Professional burnout among medical students: Systematic literature review and meta-analysis. Medical Teacher, 41(2), 172-183. https://doi.org/10.1080/0142159X.2018.1457213

[10] Almutairi, H., Alsubaiei, A., Abduljawad, S., Alshatti, A., Fekih-Romdhane, F., Husni, M., & Jahrami, H. (2022). Prevalence of burnout in medical students: A systematic review and meta-analysis. International Journal of Social Psychiatry, 68(6), 1157-1170. https://doi.org/10.1177/00207640221106691

[11] Islam, M. S., Sujan, M. S. H., Tasnim, R., Sikder, M. T., Potenza, M. N., & van Os, J. (2020). Psychological responses during the COVID-19 outbreak among university students in Bangladesh. PLOS ONE, 15(12), e0244109. https://doi.org/10.1371/journal.pone.0244109

[12] Mamun, M. A., Hossain, M. S., Siddique, A. B., Sikder, M. T., Kuss, D. J., & Griffiths, M. D. (2021). Problematic internet use in Bangladeshi students: The role of socio-demographic factors, depression, anxiety, and stress. Asian Journal of Psychiatry, 44, 48-54. https://doi.org/10.1016/j.ajp.2019.07.005

[13] Hossain, M. T., Ahammed, B., Chanda, S. K., Jahan, N., Ela, M. Z., & Islam, M. N. (2020). Social and electronic media exposure and generalized anxiety disorder among people during COVID-19 outbreak in Bangladesh: A preliminary observation. PLOS ONE, 15(9), e0238974. https://doi.org/10.1371/journal.pone.0238974

[14] Faisal, R. A., Jobe, M. C., Ahmed, O., & Sharker, T. (2022). Mental health status, anxiety, and depression levels of Bangladeshi university students during the COVID-19 pandemic. International Journal of Mental Health and Addiction, 20(3), 1500-1515. https://doi.org/10.1007/s11469-020-00458-y

[15] Salmela-Aro, K., Tolvanen, A., & Nurmi, J. E. (2009). Achievement strategies during university studies predict early career burnout and engagement. Journal of Vocational Behavior, 75(2), 162-172. https://doi.org/10.1016/j.jvb.2009.03.009

[16] Manzano-Garcia, G., & Ayala-Calvo, J. C. (2013). New perspectives: Towards an integration of the concept "burnout" and its explanatory models. Anales de Psicologia, 29(3), 800-809. https://doi.org/10.6018/analesps.29.3.161241

[17] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: Data mining, inference, and prediction (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

[18] Bzdok, D., & Meyer-Lindenberg, A. (2018). Machine learning for precision psychiatry: Opportunities and challenges. Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, 3(3), 223-230. https://doi.org/10.1016/j.bpsc.2017.11.007

[19] Iatrellis, O., Savvas, I. K., Fitsilis, P., & Gerogiannis, V. C. (2021). A two-phase machine learning approach for predicting student outcomes. Education and Information Technologies, 26(1), 69-88. https://doi.org/10.1007/s10639-020-10260-x

[20] Namoun, A., & Alshanqiti, A. (2021). Predicting student performance using data mining and learning analytics techniques: A systematic review. Applied Sciences, 11(1), 237. https://doi.org/10.3390/app11010237

[21] Gunning, D., Stefik, M., Choi, J., Miller, T., Stumpf, S., & Yang, G. Z. (2019). XAI - Explainable artificial intelligence. Science Robotics, 4(37), eaay7120. https://doi.org/10.1126/scirobotics.aay7120

[22] Molnar, C. (2022). Interpretable machine learning: A guide for making black box models explainable (2nd ed.). Independently Published.

[23] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

[24] Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20(177), 1-81.

[25] Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Kipnis, M. H. (2018). Ensuring fairness in machine learning to advance health equity. Annals of Internal Medicine, 169(12), 866-872. https://doi.org/10.7326/M18-1990

[26] Creswell, J. W., & Plano Clark, V. L. (2018). Designing and conducting mixed methods research (3rd ed.). SAGE Publications.

[27] Tashakkori, A., & Teddlie, C. (2010). SAGE handbook of mixed methods in social & behavioral research (2nd ed.). SAGE Publications. https://doi.org/10.4135/9781506335193

[28] Denzin, N. K. (1978). The research act: A theoretical introduction to sociological methods (2nd ed.). McGraw-Hill.

[29] Johnson, R. B., Onwuegbuzie, A. J., & Turner, L. A. (2007). Toward a definition of mixed methods research. Journal of Mixed Methods Research, 1(2), 112-133. https://doi.org/10.1177/1558689806298224

[30] Hossen, M. A., Ali, S., & Mamun, M. A. (2023). Psychological distress and its associated factors among university students in Bangladesh: A multi-institutional study. Journal of Affective Disorders Reports, 11, 100452. https://doi.org/10.1016/j.jadr.2022.100452

[31] Nayan, M. I. H., Uddin, M. S. G., Hossain, M. I., Alam, M. M., Zinnia, M. A., Haq, I., ... Methun, M. I. H. (2022). Comparison of the performance of machine learning-based algorithms for predicting depression and anxiety among University Students in Bangladesh: A result of the first wave of the COVID-19 pandemic. Asian Journal of Social Health and Behavior, 5(2), 75-84. https://doi.org/10.4103/shb.shb_38_22

[32] Hobfoll, S. E. (1989). Conservation of resources: A new attempt at conceptualizing stress. American Psychologist, 44(3), 513-524. https://doi.org/10.1037/0003-066X.44.3.513

[33] Salmela-Aro, K., & Upadyaya, K. (2014). School burnout and engagement in the context of demands-resources model. British Journal of Educational Psychology, 84(1), 137-151. https://doi.org/10.1111/bjep.12018

[34] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785

[35] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146-3154.

[36] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31, 6638-6648.

[37] Hobfoll, S. E., Halbesleben, J., Neveu, J. P., & Westman, M. (2018). Conservation of resources in the organizational context: The reality of resources and their consequences. Annual Review of Organizational Psychology and Organizational Behavior, 5, 103-128. https://doi.org/10.1146/annurev-orgpsych-032117-104640

[38] Bakker, A. B., & Demerouti, E. (2007). The job demands-resources model: State of the art. Journal of Managerial Psychology, 22(3), 309-328. https://doi.org/10.1108/02683940710733115

[39] Bakker, A. B., & Demerouti, E. (2017). Job demands-resources theory: Taking stock and looking forward. Journal of Occupational Health Psychology, 22(3), 273-285. https://doi.org/10.1037/ocp0000056

[40] Ryan, R. M., & Deci, E. L. (2017). Self-determination theory: Basic psychological needs in motivation, development, and wellness. Guilford Publications. https://doi.org/10.1521/978.14625/28806

[41] Deci, E. L., & Ryan, R. M. (2000). The "what" and "why" of goal pursuits: Human needs and the self-determination of behavior. Psychological Inquiry, 11(4), 227-268. https://doi.org/10.1207/S15327965PLI1104_01

[42] Steyerberg, E. W., & Harrell, F. E. (2016). Prediction models need appropriate internal, internal-external, and external validation. Journal of Clinical Epidemiology, 69, 245-247. https://doi.org/10.1016/j.jclinepi.2015.04.005

[43] Freudenberger, H. J. (1974). Staff burn-out. Journal of Social Issues, 30(1), 159-165. https://doi.org/10.1111/j.1540-4560.1974.tb00706.x

[44] Maslach, C., & Jackson, S. E. (1981). The measurement of experienced burnout. Journal of Occupational Behavior, 2(2), 99-113. https://doi.org/10.1002/job.4030020205

[45] Schaufeli, W. B., Bakker, A. B., & Van Rhenen, W. (2009). How changes in job demands and resources predict burnout, work engagement, and sickness absenteeism. Journal of Organizational Behavior, 30(7), 893-917. https://doi.org/10.1002/job.595

[46] Fazullina, G., Sagitova, R., & Khakimova, R. (2020). Burnout syndrome among university students. Revista Inclusiones, 7, 223-233.

[47] Robins, T. G., Roberts, R. M., & Sarris, A. (2018). Burnout and engagement in health profession students. Nurse Education Today, 69, 100-104. https://doi.org/10.1016/j.nedt.2018.07.004

[48] Wiese, C. W., Tay, L., Thoresen, C. J., & Kaplan, S. A. (2018). A meta-analysis of burnout and job performance: Theoretical extensions and methodological implications. Journal of Applied Psychology, 103(11), 1152-1172. https://doi.org/10.1037/apl0000329

[49] Lin, S. H., & Huang, Y. C. (2014). Life stress and academic burnout. Active Learning in Higher Education, 15(1), 77-90. https://doi.org/10.1177/1469787413514651

[50] Walburg, V. (2014). Burnout among high school students: A literature review. Children and Youth Services Review, 42, 28-33. https://doi.org/10.1016/j.childyouth.2014.03.020

[51] Koutsimani, P., Montgomery, A., & Georganta, K. (2019). The relationship between burnout, depression, and anxiety: A systematic review and meta-analysis. Frontiers in Psychology, 10, 284. https://doi.org/10.3389/fpsyg.2019.00284

[52] Riaz, M., Ali, M., & Tariq, A. (2021). Psychological burden and coping strategies among university students during times of systemic crisis. International Journal of Mental Health Systems, 15(1), 1-9. https://doi.org/10.1186/s13033-021-00468-2

[53] Shahidi, S., Akbari, H., & Zargar, F. (2023). The role of anxiety sensitivity in academic burnout among university students. Current Psychology, 42(8), 6891-6900. https://doi.org/10.1007/s12144-021-02014-9

[54] Rahmati, Z. (2015). The study of academic burnout in students with high and low self-efficacy. Procedia - Social and Behavioral Sciences, 171, 49-55. https://doi.org/10.1016/j.sbspro.2015.01.087

[55] Jackson, E. R., Shanafelt, T. D., Hasan, O., Satele, D. V., & Dyrbye, L. N. (2016). Burnout and alcohol abuse/dependence among US medical students. Academic Medicine, 91(9), 1251-1256. https://doi.org/10.1097/ACM.0000000000001138

[56] Dyrbye, L. N., Thomas, M. R., Massie, F. S., Power, D. V., Eacker, A., Harper, W., ... & Shanafelt, T. D. (2008). Burnout and suicidal ideation among US medical students. Annals of Internal Medicine, 149(5), 334-341. https://doi.org/10.7326/0003-4819-149-5-200809020-00008

[57] Silva, R. M. D., Lopes, A. A. F., & Ribeiro, H. K. P. (2022). Burnout syndrome among Brazilian university students. Revista Brasileira de Enfermagem, 75(4), e20210470. https://doi.org/10.1590/0034-7167-2021-0470

[58] Islam, M. A., Barna, S. D., Raihan, H., Khan, M. N. A., & Hossain, M. T. (2020). Depression and anxiety among university students during the COVID-19 pandemic in Bangladesh: A web-based cross-sectional survey. PLOS ONE, 15(8), e0238162. https://doi.org/10.1371/journal.pone.0238162

[59] Dahlin, M. E., & Runeson, B. (2007). Burnout and psychiatric morbidity among medical students entering clinical training. BMC Medical Education, 7(1), 1-8. https://doi.org/10.1186/1472-6920-7-6

[60] Ali, S., Hossain, M. T., Islam, M. A., & Barna, S. D. (2021). Prevalence of depression, anxiety, and stress among university students in Bangladesh: A systematic review and meta-analysis. Journal of Affective Disorders, 280, 25-34. https://doi.org/10.1016/j.jad.2020.11.054

[61] Celik, E., & Yildirim, T. (2022). Academic burnout among Turkish university students during the COVID-19 pandemic. International Journal of Educational Research Open, 3, 100147. https://doi.org/10.1016/j.ijedro.2022.100147

[62] Almeida, G. C., Souza, H. R., Almeida, P. C., Almeida, B. C., & Almeida, G. H. (2021). The prevalence of burnout syndrome in medical students. Archives of Clinical Psychiatry, 48(1), 40-47. https://doi.org/10.1590/0101-60830000000282

[63] Bask, M., & Salmela-Aro, K. (2013). Burned out to drop out: Exploring the relationship between school burnout and school dropout. European Journal of Psychology of Education, 28(2), 511-528. https://doi.org/10.1007/s10212-012-0126-5

[64] Bianchi, R., Schonfeld, I. S., & Laurent, E. (2015). Burnout-depression overlap: A review. Clinical Psychology Review, 36, 28-41. https://doi.org/10.1016/j.cpr.2015.01.004

[65] Robotham, D., & Julian, C. (2006). Stress and the higher education student: A critical review of the literature. Journal of Further and Higher Education, 30(2), 107-117. https://doi.org/10.1080/03098770600617562

[66] Alarcon, G., Eschleman, K. J., & Bowling, N. A. (2009). Relationships between personality variables and burnout: A meta-analysis. Work & Stress, 23(3), 244-263. https://doi.org/10.1080/02678370903282600

[67] Richardson, T., Elliott, P., & Roberts, R. (2017). Relationship between loneliness and mental health in students. Journal of Public Mental Health, 16(2), 48-54. https://doi.org/10.1108/JPMH-03-2016-0013

[68] Walsemann, K. M., Gee, G. C., & Gentile, D. (2015). Sick of our loans: Student borrowing and the mental health of young adults in the United States. Social Science & Medicine, 124, 85-93. https://doi.org/10.1016/j.socscimed.2014.11.027

[69] Hershner, S. D., & Chervin, R. D. (2014). Causes and consequences of sleepiness among college students. Nature and Science of Sleep, 6, 73-84. https://doi.org/10.2147/NSS.S62907

[70] Lund, H. G., Reider, B. D., Whiting, A. B., & Prichard, J. R. (2010). Sleep patterns and predictors of disturbed sleep in a large population of college students. Journal of Adolescent Health, 46(2), 124-132. https://doi.org/10.1016/j.jadohealth.2009.06.016

[71] Tsanas, A., Little, M. A., Fox, C., & Ramchurn, I. (2016). Objective automatic assessment of sleep quality using wearable sensors and non-linear dynamics. IEEE Transactions on Biomedical Engineering, 63(4), 758-765. https://doi.org/10.1109/TBME.2015.2476832

[72] Ahrberg, K., Dresler, M., Niedermaier, S., Steiger, A., & Genzel, L. (2012). The interaction between sleep quality and academic performance. Journal of Psychiatric Research, 46(12), 1618-1622. https://doi.org/10.1016/j.jpsychires.2012.09.008

[73] Romero-Blanco, C., Rodriguez-Almagro, J., Onieva-Zafra, M. D., Parra-Fernandez, M. L., Prado-Laguna, M. D. C., & Hernandez-Martinez, A. (2020). Sleep pattern changes in nursing students during the COVID-19 lockdown. International Journal of Environmental Research and Public Health, 17(14), 5222. https://doi.org/10.3390/ijerph17145222

[74] Naczenski, L. M., Vries, J. D., van Hooff, M. L., & Kompier, M. A. (2017). Systematic review of the association between physical activity and burnout. Journal of Occupational Health, 59(6), 477-494. https://doi.org/10.1539/joh.17-0050-RA

[75] Woods, H. C., & Scott, H. (2016). #Sleepyteens: Social media use in adolescence is associated with poor sleep quality, anxiety, depression and low self-esteem. Journal of Adolescence, 51, 41-49. https://doi.org/10.1016/j.adolescence.2016.05.008

[76] Primack, B. A., Shensa, A., Escobar-Viera, C. G., Barrett, E. L., Sidani, J. E., Colditz, J. B., & James, A. E. (2017). Use of multiple social media platforms and symptoms of depression and anxiety. Computers in Human Behavior, 69, 1-9. https://doi.org/10.1016/j.chb.2016.11.013

[77] Plackett, R., Blythe, A., Copello, A., & Mars, B. (2020). The impact of social media use on adolescent mental health: A structured review. Journal of Adolescent Health, 67(1), 12-21. https://doi.org/10.1016/j.jadohealth.2020.01.015

[78] Keles, B., McCrae, N., & Grealish, A. (2020). A systematic review: the influence of social media on depression, anxiety and psychological distress in adolescents. International Journal of Adolescence and Youth, 25(1), 79-93. https://doi.org/10.1080/02673843.2019.1590851

[79] Fang, J., Wang, X., Wen, Z., & Zhou, J. (2022). Fear of missing out and problematic social media use as mediators between emotional support from social media and phubbing behavior. Addictive Behaviors, 107, 106430. https://doi.org/10.1016/j.addbeh.2020.106430

[80] Ng, A. W., Ye, Y. C., & Lee, S. K. (2022). Social media addiction and its impact on college students' mental health: A predictive modeling approach. Journal of Educational Computing Research, 60(3), 675-698. https://doi.org/10.1177/07356331211041926

[81] Purvanova, R. K., & Muros, J. P. (2010). Gender differences in burnout: A meta-analysis. Journal of Vocational Behavior, 77(2), 168-185. https://doi.org/10.1016/j.jvb.2010.04.006

[82] Kim, B., Jee, S., Lee, J., An, S., & Lee, S. M. (2018). Relationships between social support and student burnout: A meta-analytic approach. Stress and Health, 34(1), 127-134. https://doi.org/10.1002/smi.2771

[83] Baker, R. S. J. D., & Inventado, P. S. (2014). Educational data mining and learning analytics. In Learning analytics (pp. 61-75). Springer. https://doi.org/10.1007/978-1-4614-3305-7_4

[84] Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 10(3), e1355. https://doi.org/10.1002/widm.1355

[85] Shahiri, A. M., Husain, W., & Rashid, N. A. (2015). A review on predicting student performance using data mining techniques. Procedia Computer Science, 72, 414-422. https://doi.org/10.1016/j.procs.2015.12.157

[86] Alhazmi, E., & Sheneamer, A. (2023). Early predicting of students performance in higher education. IEEE Access, 11, 27579-27589. https://doi.org/10.1109/ACCESS.2023.3250702

[87] Priya, A., Garg, S., & Tigga, N. P. (2020). Predicting anxiety, depression and stress in modern life using machine learning algorithms. Procedia Computer Science, 167, 1258-1267. https://doi.org/10.1016/j.procs.2020.03.442

[88] Zheng, X., Chen, Y., & Liu, Y. (2021). Machine learning algorithms for predicting depression among university students. Computers in Human Behavior, 120, 106752. https://doi.org/10.1016/j.chb.2021.106752

[89] van der Ploeg, T., Austin, P. C., & Steyerberg, E. W. (2014). Modern modelling techniques are data hungry: A simulation study for predicting dichotomous endpoints. BMC Medical Research Methodology, 14(1), 1-11. https://doi.org/10.1186/1471-2288-14-137

[90] Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In Proceedings of the 23rd International Conference on Machine Learning (pp. 233-240). ACM. https://doi.org/10.1145/1143844.1143874

[91] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1), 56-67. https://doi.org/10.1038/s42256-019-0138-9

[92] Tsiakmaki, M., Kostopoulos, G., Kotsiantis, S., & Ragos, O. (2020). Implementing AutoML in educational data mining for prediction tasks. Applied Sciences, 10(1), 90. https://doi.org/10.3390/app10010090

[93] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

[94] Strobl, C., Boulesteix, A. L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. BMC Bioinformatics, 8(1), 1-21. https://doi.org/10.1186/1471-2105-8-25

[95] Nicodemus, K. K., Malley, J. D., Strobl, C., & Ziegler, A. (2010). The behaviour of random forest permutation-based variable importance measures under predictor correlation. BMC Bioinformatics, 11(1), 1-13. https://doi.org/10.1186/1471-2105-11-110

[96] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232. https://doi.org/10.1214/aos/1013203451

[97] Hooker, G., Mentch, L., & Zhou, S. (2021). Unrestricted permutation forces extrapolation: Variable importance requires at least one more model, or there is no free variable importance. Statistics and Computing, 31(5), 1-11. https://doi.org/10.1007/s11222-021-10057-z

[98] Greenwell, B. M. (2017). pdp: An R package for constructing partial dependence plots. The R Journal, 9(1), 421-436. https://doi.org/10.32614/RJ-2017-016

[99] Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. Journal of Computational and Graphical Statistics, 24(1), 44-65. https://doi.org/10.1080/10618600.2014.907095

[100] Sarkar, S., Ray, A., & Sharma, M. (2022). Explainable AI in healthcare and medicine: A systematic review. Journal of Medical Systems, 46(12), 85. https://doi.org/10.1007/s10916-022-01869-7

[101] Rastrollo-Guerrero, J. L., Gómez-Pulido, J. A., & Durán-Domínguez, A. (2020). Analyzing and predicting students' performance using machine learning: A systematic review. Applied Sciences, 10(6), 2023. https://doi.org/10.3390/app10062023

[102] Braun, V., & Clarke, V. (2021). One size fits all? What counts as quality practice in (reflexive) thematic analysis? Qualitative Research in Psychology, 18(3), 328-352. https://doi.org/10.1080/14780887.2020.1769238

[103] Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. Qualitative Research in Psychology, 3(2), 77-101. https://doi.org/10.1191/1478088706qp063oa

[104] Guest, G., Bunce, A., & Johnson, L. (2006). How many interviews are enough? An experiment with data saturation and variability. Field Methods, 18(1), 59-82. https://doi.org/10.1177/1525822X05279903

[105] Kristensen, T. S., Borritz, M., Villadsen, E., & Christensen, K. B. (2005). The Copenhagen Burnout Inventory: A new tool for the assessment of burnout. Work & Stress, 19(3), 192-207. https://doi.org/10.1080/02678370500297720

[106] Rohland, B. M., Kruse, G. R., & Rohrer, J. E. (2004). Validation of a single-item measure of burnout against the Maslach Burnout Inventory among physicians. Stress and Health, 20(2), 75-79. https://doi.org/10.1002/smi.1003

[107] West, C. P., Dyrbye, L. N., Sloan, J. A., & Shanafelt, T. D. (2009). Single item measures of emotional exhaustion and depersonalization are useful for assessing burnout in medical professionals. Journal of General Internal Medicine, 24(12), 1318-1321. https://doi.org/10.1007/s11606-009-1129-z

[108] Elo, A. L., Leppanen, A., & Jahkola, A. (2003). Validity of a single-item measure of stress symptoms. Scandinavian Journal of Work, Environment & Health, 29(6), 444-451. https://doi.org/10.5271/sjweh.752

[109] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

[110] Bzdok, D., Altman, N., & Krzywinski, M. (2018). Statistics versus machine learning. Nature Methods, 15(4), 233-234. https://doi.org/10.1038/nmeth.4642

[111] Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. Biometrics, 33(1), 159-174. https://doi.org/10.2307/2529310

[112] Farmer, T., Robinson, K., Elliott, S. J., & Eyles, J. (2006). Developing and implementing a triangulation protocol for qualitative health research. Qualitative Health Research, 16(3), 377-394. https://doi.org/10.1177/1049732305285708

[113] Tukey, J. W. (1977). Exploratory data analysis. Addison-Wesley.

[114] Hirshkowitz, M., Whiton, K., Albert, S. M., Alessi, C., Bruni, O., DonCarlos, L., Hazen, N., Herman, J., Katz, E. S., Kheirandish-Gozal, L., Neubauer, D. N., O'Donnell, A. E., Ohayon, M., Peever, J., Rawding, R., Sachdeva, R. C., Setters, B., Vitiello, M. V., Cates, J. C., & Adams Hillard, P. J. (2015). National Sleep Foundation's sleep time duration recommendations: Methodology and results summary. Sleep Health, 1(1), 40-43. https://doi.org/10.1016/j.sleh.2014.12.010

[115] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284. https://doi.org/10.1109/TKDE.2008.239

[116] Ghasemi, A., & Zahediasl, S. (2012). Normality tests for statistical analysis: A guide for non-statisticians. International Journal of Endocrinology and Metabolism, 10(2), 186-191. https://doi.org/10.5812/ijem.3505

[117] Faul, F., Erdfelder, E., Lang, A. G., & Buchner, A. (2007). G*Power 3: A flexible statistical power analysis program for the social, behavioral, and biomedical sciences. Behavior Research Methods, 39(2), 175-191. https://doi.org/10.3758/BF03193146

[118] Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum Associates.

[119] O'Brien, R. M. (2007). A caution regarding rules of thumb for variance inflation factors. Quality & Quantity, 41(5), 673-690. https://doi.org/10.1007/s11135-006-9018-6

[120] Strobl, C., Malley, J., & Tutz, G. (2009). An introduction to recursive partitioning: Rationale, application, and characteristics of classification and regression trees, bagging, and random forests. Psychological Methods, 14(4), 323-348. https://doi.org/10.1037/a0016973

[121] Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874. https://doi.org/10.1016/j.patrec.2005.10.010

[122] Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5), 206-215. https://doi.org/10.1038/s42256-019-0048-x

[123] Mamun, M. A., & Griffiths, M. D. (2019). The psychological impact of extreme weather events and natural disasters on Bangladeshi students. Psychiatry Research, 281, 112574. https://doi.org/10.1016/j.psychres.2019.112574

[124] Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): The TRIPOD statement. BMJ, 350, g7594. https://doi.org/10.1136/bmj.g7594

[125] Moons, K. G., Altman, D. G., Reitsma, J. B., Ioannidis, J. P., Macaskill, P., Steyerberg, E. W., ... & Collins, G. S. (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): Explanation and elaboration. Annals of Internal Medicine, 162(1), W1-W73. https://doi.org/10.7326/M14-0698
