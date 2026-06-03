# Predicting Poor Clinical Outcome in Paediatric Sepsis Using Point-of-Care Features

**Md Rafin Rahman**[1], **Abdallah Tasawar Khan**[2], **Jafren Iqbal Rose**[3] <br>
[1] institute for developing Science and Health initiatives (ideSHi), Dhaka, Bangladesh
[2] Kurmitola General Hospital, Dhaka, Bangladesh
[3] Shaheed Monsur Ali Medical College & Hospital

Correspondence: mrahman@ideshi.org

**Contact:** mrahman@ideshi.org

---

## Overview

This repository contains the complete analysis code for the manuscript:

> **Predicting Poor Clinical Outcome in Children with Suspected Sepsis in a Low-Resource Setting Using Point-of-Care Features: Model Development and Evaluation on a Ugandan LMIC Cohort**
> Md Rafin Rahman. *Submitted for publication, 2026.*

The study develops and evaluates a machine learning model for predicting a composite poor clinical outcome (in-hospital death or length of stay exceeding five days) in children under five years admitted with suspected sepsis in a sub-Saharan African low-resource setting. The model is constrained to point-of-care clinical features only — no laboratory tests, no imaging — collectable by a trained nurse or community health worker with a thermometer, pulse oximeter, and MUAC tape.

---

## Repository Structure

```
├── 01_preprocess.py          # Data cleaning, feature engineering, encoding, imputation
├── 02_model.py               # Baseline model training and evaluation (4 algorithms)
├── 03_tune.py                # Hyperparameter optimisation with Optuna (TPE sampler)
├── 04_shap.py                # SHAP explainability analysis (LinearExplainer)
├── data/
│   ├── feature_list.json     # Final list of 45 selected features
│   └── processed.csv         # Preprocessed dataset (NOT included — see Data Access below)
├── outputs/
│   ├── tuned_results.json    # Final model performance metrics (holdout test set)
│   ├── final_calibrated_stats.json  # Platt-calibrated model statistics with 95% CI
│   ├── shap_importance.csv   # SHAP feature importance ranking
│   └── lr_coefficients.csv   # Standardised logistic regression coefficients
└── figures/
    ├── tuned_model_evaluation.png   # ROC, PR, calibration, threshold analysis
    ├── shap_analysis.png            # SHAP beeswarm, dependence, waterfall plots
    └── calibration_comparison.png   # Pre/post Platt scaling calibration comparison
```

---

## Data Access

**The dataset is not included in this repository.**

The analysis uses the **2024 Pediatric Sepsis Data Challenge Synthetic Training Dataset**, distributed by the Pediatric Sepsis Data CoLaboratory (Sepsis CoLab) through the Borealis Dataverse at the University of British Columbia.

- **DOI:** [10.5683/SP3/TFAV36](https://doi.org/10.5683/SP3/TFAV36)
- **Access:** Free membership registration at [https://borealisdata.ca/dataverse/Pedi_SepsisCoLab](https://borealisdata.ca/dataverse/Pedi_SepsisCoLab)

The synthetic dataset was generated from a real-world prospective cohort of 3,837 children admitted with suspected sepsis across six Ugandan hospitals (2017–2020), described in:

> Huxford C, Rafiei A, Nguyen V, et al. The 2024 Pediatric Sepsis Challenge: Predicting In-Hospital Mortality in Children With Suspected Sepsis in Uganda. *Pediatr Crit Care Med.* 2024. DOI: [10.1097/PCC.0000000000003556](https://doi.org/10.1097/PCC.0000000000003556)

Once you have downloaded `PSDC_SyntheticTrainingData_Dataset.csv` from Borealis, place it at `data/raw.csv` before running the pipeline.

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn optuna scipy imbalanced-learn pillow
```

Python 3.10 or above is required. All packages are available via pip. No GPU is required.

---

## How to Reproduce the Analysis

Run the scripts in order:

```bash
# Step 1: Preprocess the raw dataset
python 01_preprocess.py

# Step 2: Train and evaluate baseline models
python 02_model.py

# Step 3: Hyperparameter tuning with Optuna (30 trials per model)
python 03_tune.py

# Step 4: SHAP explainability analysis on the best model
python 04_shap.py
```

Each script saves its outputs to the `outputs/` and `figures/` directories. Total runtime on a standard laptop (8 CPU cores) is approximately 25 to 40 minutes, dominated by Steps 2 and 3.

---

## Key Results

| Model | CV AUC-ROC | Test AUC-ROC | AUC-PR | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression* | 0.691 | 0.686† | 0.557† | 0.571‡ | 0.769‡ | 0.199† |
| Random Forest | 0.697 | 0.682 | 0.577 | 0.423 | 0.825 | 0.209 |
| XGBoost | 0.690 | 0.684 | 0.570 | 0.445 | 0.797 | 0.214 |
| LightGBM | 0.689 | 0.680 | 0.574 | 0.467 | 0.789 | 0.214 |

\* Best model by AUC-ROC. CV = five-fold cross-validation on development set.
† After post-hoc Platt scaling calibration. Bootstrap 95% CI: AUC-ROC 0.637–0.734; AUC-PR 0.486–0.637; Brier 0.182–0.217.
‡ At Youden J-optimised threshold of 0.33. Hosmer-Lemeshow p = 0.486; E/O ratio = 1.027.

**Top 5 predictors by mean absolute SHAP value:**
1. Severe Acute Malnutrition — MUAC <115 mm (0.193)
2. Weight, kg (0.128)
3. WHO Tachypnoea (0.111)
4. Moderate Acute Malnutrition — MUAC 115–125 mm (0.107)
5. MUAC, mm continuous (0.096)

---

## Outcome Definition

The composite outcome was defined as **in-hospital death OR length of stay exceeding five days**, consistent with published approaches to operationalising treatment failure and poor sepsis outcome in LMIC settings where 48-hour reassessment data are not routinely available.

- Total records: 2,685
- Poor outcome (positive class): 911 (33.9%)
- Good outcome (negative class): 1,774 (66.1%)
- In-hospital deaths (component): 119 (4.4%)

---

## Feature Set

Forty-five point-of-care clinical features were selected, collectable without laboratory or imaging resources. Categories include:

- **Continuous vitals:** age, weight, MUAC, heart rate, respiratory rate, temperature, SpO₂, systolic BP, illness duration
- **Clinical signs:** respiratory distress, capillary refill
- **Symptoms:** cough, fever, vomiting, altered consciousness, seizures, others
- **Comorbidities:** asthma, cardiac disease, sickle cell, tuberculosis
- **History:** prior antibiotics, malaria RDT, HIV status, vaccination, TB contact
- **Engineered features:** WHO tachypnoea flag, severe hypoxia (SpO₂ <90%), SAM by MUAC (<115 mm), MAM by MUAC (115–125 mm), Blantyre Coma Scale total

Full feature descriptions are in `data/feature_list.json` and Section 2.3 of the manuscript.

---

## Calibration

The logistic regression model was found to be systematically miscalibrated prior to post-hoc correction (Hosmer-Lemeshow chi-squared = 57.77, p <0.001; E/O = 0.713), a known consequence of class-weight balancing. Post-hoc Platt scaling was applied by subdividing the development set into an 80% training subset and a 20% calibration subset, fitting a sigmoid calibrator on the calibration fold, and evaluating on the held-out test set. After calibration, Hosmer-Lemeshow p = 0.486 and E/O = 1.027.

---

## Citation

If you use this code, please cite the manuscript:

```
Rahman MR. Predicting Poor Clinical Outcome in Children with Suspected Sepsis in a
Low-Resource Setting Using Point-of-Care Features: Model Development and Evaluation
on a Ugandan LMIC Cohort. Submitted for publication, 2026.
ideSHi, Dhaka, Bangladesh. Contact: mrahman@ideshi.org
```

Please also cite the original dataset:

```
Huxford C, Rafiei A, Nguyen V, et al. The 2024 Pediatric Sepsis Challenge: Predicting
In-Hospital Mortality in Children With Suspected Sepsis in Uganda.
Pediatr Crit Care Med. 2024. DOI: 10.1097/PCC.0000000000003556
```

---

## Limitations

- The dataset is synthetically generated from real patient data to reduce re-identification risk. While the CoLab has documented statistical fidelity to the original distributions, distributional artefacts from the synthesis process cannot be excluded.
- The model was developed on a general sepsis cohort in Uganda. Malaria RDT positivity is a high-prevalence variable in this setting that shows no group-level discrimination (31.4% vs 31.1%) but contributes SHAP weight through collinearity. This feature should be excluded or recalibrated when applying the model in non-malaria-endemic settings such as Bangladesh.
- No external validation has been performed. Prospective validation on a Bangladeshi clinical cohort is the recommended next step.

---

## Licence

Code in this repository is released under the **MIT Licence**. The dataset is governed by the Sepsis CoLab data sharing agreement and is not redistributed here.

---

*Institute for Developing Science and Health Initiatives (ideSHi) | Dhaka, Bangladesh | June 2026*
