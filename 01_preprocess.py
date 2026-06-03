"""
Step 01: Preprocessing
Project: Predicting Poor Clinical Outcome in Pediatric Sepsis
         Using Point-of-Care Features in a Low-Resource Setting
Author:  Md Rafin Rahman, ideSHi
Dataset: PSDC Synthetic Training Dataset (Borealis / Sepsis CoLab)
"""

import pandas as pd
import numpy as np
import json

# ── 1. LOAD ───────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/project/data/raw.csv')
print(f"Raw shape: {df.shape}")

# ── 2. EXCLUDE BAD RECORDS ────────────────────────────────────────────
# One record has negative LOS (data entry error)
df = df[df['lengthadm'] >= 0].copy()
print(f"After excluding negative LOS: {df.shape}")

# ── 3. DEFINE OUTCOME ─────────────────────────────────────────────────
# Composite poor outcome: in-hospital death OR prolonged stay (LOS > 5 days)
# Rationale: median LOS = 4 days; 75th percentile = 6 days.
# LOS > 5 days captures cases requiring escalated care or treatment change,
# consistent with published LMIC pneumonia/sepsis outcome literature
# (Amare et al. PLOS ONE 2023; Fox et al. Clin Infect Dis 2013).
df['poor_outcome'] = ((df['inhospital_mortality'] == 1) |
                      (df['lengthadm'] > 5)).astype(int)

print(f"\nOutcome distribution:")
print(f"  Poor outcome (LOS>5 OR death): {df['poor_outcome'].sum()} / {len(df)} "
      f"({100*df['poor_outcome'].mean():.1f}%)")
print(f"  Good outcome:                  {(df['poor_outcome']==0).sum()} / {len(df)} "
      f"({100*(df['poor_outcome']==0).mean():.1f}%)")
print(f"  Deaths included in outcome:    {df['inhospital_mortality'].sum()}")

# ── 4. SELECT FEATURES ────────────────────────────────────────────────
# Strictly point-of-care: no laboratory tests, no imaging.
# All features collectable by a trained nurse or community health worker
# with a pulse oximeter, thermometer, and MUAC tape.
# Intervention variables (admitabx) excluded per data dictionary guidance.
# Lab variables (lactate, hematocrit, glucose) excluded from main model.

CONTINUOUS = [
    'agecalc_adm',          # Age at admission (months)
    'weight_kg_adm',        # Weight (kg)
    'muac_mm_adm',          # MUAC (mm) - proxy for acute malnutrition
    'hr_bpm_adm',           # Heart rate
    'rr_brpm_app_adm',      # Respiratory rate
    'temp_c_adm',           # Axillary temperature
    'spo2site1_pc_oxi_adm', # SpO2 measure 1
    'badhealthduration_adm',# Duration of illness before admission
    'sysbp_mmhg_adm',       # Systolic BP
]

BINARY_YESNO = [
    'priorweekabx_adm',     # Antibiotics in prior week
    'prioryearcough_adm',   # Prior year cough / breathing difficulty
    'tbcontact_adm',        # TB contact in household
    'malariastatuspos_adm', # Malaria RDT positive
    'prioryearwheeze_adm',  # Prior year wheeze
    'diarrheaoften_adm',    # Frequent diarrhea
]

SYMPTOMS = [
    'symptoms_adm___2',     # Cough < 14 days
    'symptoms_adm___3',     # Cough > 14 days
    'symptoms_adm___4',     # Diarrhea < 14 days
    'symptoms_adm___6',     # Fever < 7 days
    'symptoms_adm___7',     # Fever > 7 days
    'symptoms_adm___8',     # Vomiting everything
    'symptoms_adm___9',     # Abnormally sleepy
    'symptoms_adm___10',    # Swelling of both feet
    'symptoms_adm___14',    # Seizures / convulsions
    'symptoms_adm___15',    # Coma
]

COMORBIDITIES = [
    'comorbidity_adm___1',  # Asthma / reactive airway disease
    'comorbidity_adm___3',  # Cardiac disease
    'comorbidity_adm___5',  # Sickle cell disease
    'comorbidity_adm___6',  # Tuberculosis
    'comorbidity_adm___11', # No comorbidity
]

CATEGORICAL_MULTI = [
    'sex_adm',              # Sex
    'feedingstatus_adm',    # Feeding status (3 levels)
    'bcseye_adm',           # BCS Eye (3 levels)
    'bcsmotor_adm',         # BCS Motor (3 levels)
    'bcsverbal_adm',        # BCS Verbal (3 levels)
    'hivstatus_adm',        # HIV status
    'vaccpneumoc_adm',      # Pneumococcal vaccination
    'oxygenavail_adm',      # Oxygen availability status
]

ALL_FEATURES = (CONTINUOUS + ['respdistress_adm', 'caprefill_adm'] +
                BINARY_YESNO + SYMPTOMS + COMORBIDITIES + CATEGORICAL_MULTI)

# Convert StringArray columns to object dtype before selection to avoid encoding issues
string_cols = ['respdistress_adm', 'caprefill_adm', 'hivstatus_adm']
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(object)

df_model = df[ALL_FEATURES + ['poor_outcome', 'studyid_adm']].copy()
print(f"\nFeature set: {len(ALL_FEATURES)} features selected")
print(f"  Continuous:           {len(CONTINUOUS)}")
print(f"  Binary (Yes/No):      {len(BINARY_YESNO)}")
print(f"  Symptoms (Checked):   {len(SYMPTOMS)}")
print(f"  Comorbidities:        {len(COMORBIDITIES)}")
print(f"  Categorical (multi):  {len(CATEGORICAL_MULTI)}")

# ── 5. ENGINEER DERIVED FEATURES ──────────────────────────────────────
# Age-specific tachypnea flag (WHO IMCI thresholds)
def tachypnea_flag(row):
    rr  = row['rr_brpm_app_adm']
    age = row['agecalc_adm']
    if pd.isna(rr) or pd.isna(age):
        return np.nan
    if age < 2:   return int(rr >= 60)
    elif age < 12: return int(rr >= 50)
    else:          return int(rr >= 40)

df_model['tachypnea_who'] = df_model.apply(tachypnea_flag, axis=1)

# Severe hypoxia flag (SpO2 < 90%)
df_model['severe_hypoxia'] = (df_model['spo2site1_pc_oxi_adm'] < 90).astype(float)
df_model.loc[df_model['spo2site1_pc_oxi_adm'].isna(), 'severe_hypoxia'] = np.nan

# Severe acute malnutrition by MUAC (< 115 mm, WHO threshold)
df_model['sam_muac'] = (df_model['muac_mm_adm'] < 115).astype(float)
df_model.loc[df_model['muac_mm_adm'].isna(), 'sam_muac'] = np.nan

# Moderate acute malnutrition by MUAC (115-125 mm)
df_model['mam_muac'] = ((df_model['muac_mm_adm'] >= 115) &
                         (df_model['muac_mm_adm'] < 125)).astype(float)
df_model.loc[df_model['muac_mm_adm'].isna(), 'mam_muac'] = np.nan

# BCS composite (sum of eye + motor + verbal, 0-9, lower = worse consciousness)
bcs_map_eye    = {'Fails to watch or follow': 0,
                  'Watches or follows': 2}
bcs_map_motor  = {'No response or inappropriate response': 0,
                  'Withdraws limb from painful stimulus': 1,
                  'Localizes painful stimulus': 2}
bcs_map_verbal = {'No vocal response to pain': 0,
                  'Moan or abnormal cry with pain': 1,
                  'Cries appropriately with pain, or, if verbal, speaks': 2}

df_model['bcs_eye_num']    = df_model['bcseye_adm'].map(bcs_map_eye)
df_model['bcs_motor_num']  = df_model['bcsmotor_adm'].map(bcs_map_motor)
df_model['bcs_verbal_num'] = df_model['bcsverbal_adm'].map(bcs_map_verbal)
df_model['bcs_total']      = (df_model['bcs_eye_num'] +
                               df_model['bcs_motor_num'] +
                               df_model['bcs_verbal_num'])

print(f"\nEngineered features added: tachypnea_who, severe_hypoxia, sam_muac, mam_muac, bcs_total")

# ── 6. ENCODE CATEGORICAL VARIABLES ───────────────────────────────────
# Binary Yes/No → 1/0
yesno_vars = BINARY_YESNO
for col in yesno_vars:
    if col in df_model.columns:
        df_model[col] = df_model[col].map({'Yes': 1, 'No': 0})

# respdistress and caprefill - cast to object first to avoid StringArray issues
for col in ['respdistress_adm', 'caprefill_adm']:
    df_model[col] = df_model[col].astype(object)
    df_model[col] = df_model[col].apply(
        lambda x: 1 if x == 'Yes' else (0 if x == 'No' else np.nan))

# Checked/Unchecked → 1/0
for col in SYMPTOMS + COMORBIDITIES:
    df_model[col] = df_model[col].map({'Checked': 1, 'Unchecked': 0})

# Sex
df_model['sex_adm'] = df_model['sex_adm'].map({'Male': 1, 'Female': 0})

# HIV
df_model['hivstatus_adm'] = df_model['hivstatus_adm'].apply(
    lambda x: 1 if str(x) == 'HIV positive' else (0 if str(x) == 'HIV negative' else np.nan))

# Feeding status (ordinal: well=0, poorly=1, not at all=2)
df_model['feedingstatus_adm'] = df_model['feedingstatus_adm'].map(
    {'Feeding well': 0, 'Feeding poorly': 1, 'Not feeding at all': 2})

# Pneumococcal vaccination (ordinal by dose count; Unknown → NaN)
vacc_map = {'0 doses': 0, '1 dose': 1, '2 doses': 2, '3 doses': 3, 'Unknown': np.nan}
df_model['vaccpneumoc_adm'] = df_model['vaccpneumoc_adm'].map(vacc_map)

# Oxygen availability (ordinal)
oxy_map = {
    'Oxygen available and being used': 2,
    'Oxygen available but not enough': 1,
    'Oxygen available and not being used': 0,
    'Oxygen not available': 0
}
df_model['oxygenavail_adm'] = df_model['oxygenavail_adm'].map(oxy_map)

# Duration of illness (ordinal)
dur_map = {
    'In good health prior to this illness': 0,
    '< 1 week': 1,
    '1 week - 1 month': 2,
    '1 month - 1 year': 3,
    '> 1 year': 4,
    'Unknown': np.nan
}
df_model['badhealthduration_adm'] = df_model['badhealthduration_adm'].map(dur_map)

# BCS categorical already handled via numeric above; drop original text cols
df_model = df_model.drop(columns=['bcseye_adm', 'bcsmotor_adm', 'bcsverbal_adm'])

print("\nCategorical encoding done.")

# ── 7. HANDLE MISSING VALUES ──────────────────────────────────────────
# Check remaining missingness
miss = df_model.drop(columns=['studyid_adm','poor_outcome']).isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print(f"\nMissing values after encoding:")
for col, n in miss.items():
    print(f"  {col:<35} {n:4d} ({100*n/len(df_model):.1f}%)")

# Median imputation for continuous/numeric variables with missing values
impute_continuous = CONTINUOUS + ['tachypnea_who', 'severe_hypoxia', 'sam_muac',
                                   'mam_muac', 'bcs_total', 'bcs_eye_num',
                                   'bcs_motor_num', 'bcs_verbal_num',
                                   'badhealthduration_adm', 'vaccpneumoc_adm']
for col in impute_continuous:
    if col in df_model.columns:
        n_imp = df_model[col].isna().sum()
        if n_imp > 0:
            median_val = pd.to_numeric(df_model[col], errors='coerce').median()
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(median_val)
            print(f"  Imputed {col} with median {median_val:.2f} (n={n_imp})")

# Mode imputation for binary/categorical variables
impute_categorical = (BINARY_YESNO + ['respdistress_adm', 'caprefill_adm'] +
                      SYMPTOMS + COMORBIDITIES +
                      ['sex_adm', 'feedingstatus_adm', 'hivstatus_adm', 'oxygenavail_adm'])
for col in impute_categorical:
    if col in df_model.columns:
        n_imp = df_model[col].isna().sum()
        if n_imp > 0:
            numeric_series = pd.to_numeric(df_model[col], errors='coerce')
            modes = numeric_series.mode()
            if len(modes) == 0:
                continue
            mode_val = modes[0]
            df_model[col] = numeric_series.fillna(mode_val)
            print(f"  Imputed {col} with mode {mode_val} (n={n_imp})")

# ── 8. FINAL FEATURE LIST ─────────────────────────────────────────────
FINAL_FEATURES = [c for c in df_model.columns
                  if c not in ['poor_outcome', 'studyid_adm']]

print(f"\nFinal feature count: {len(FINAL_FEATURES)}")
print(f"Final dataset shape: {df_model.shape}")
print(f"Missing values remaining: {df_model[FINAL_FEATURES].isna().sum().sum()}")

# ── 9. SAVE ───────────────────────────────────────────────────────────
df_model.to_csv('/home/claude/project/data/processed.csv', index=False)
with open('/home/claude/project/data/feature_list.json', 'w') as f:
    json.dump(FINAL_FEATURES, f, indent=2)

print("\n✓ Saved: processed.csv and feature_list.json")
print("\n=== PREPROCESSING COMPLETE ===")
print(f"Records:         {len(df_model)}")
print(f"Features:        {len(FINAL_FEATURES)}")
print(f"Poor outcome:    {df_model['poor_outcome'].sum()} ({100*df_model['poor_outcome'].mean():.1f}%)")
print(f"Good outcome:    {(df_model['poor_outcome']==0).sum()} ({100*(df_model['poor_outcome']==0).mean():.1f}%)")
