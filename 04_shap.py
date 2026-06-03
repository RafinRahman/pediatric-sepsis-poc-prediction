"""
Step 04: SHAP Analysis
Project: Predicting Poor Clinical Outcome in Pediatric Sepsis
Author:  Md Rafin Rahman, ideSHi
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── 1. LOAD AND REBUILD BEST MODEL ───────────────────────────────────
df = pd.read_csv('/home/claude/project/data/processed.csv')
with open('/home/claude/project/data/feature_list.json') as f:
    FEATURES = json.load(f)

X    = df[FEATURES].values
y    = df['poor_outcome'].values
X_df = df[FEATURES].copy()

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

X_dev_df, X_test_df = train_test_split(
    X_df, test_size=0.20, random_state=42)[0], \
    train_test_split(X_df, test_size=0.20, random_state=42)[1]

# Rebuild the scaler and model separately for SHAP (SHAP needs raw LR coefs)
scaler = StandardScaler()
X_dev_sc  = scaler.fit_transform(X_dev)
X_test_sc = scaler.transform(X_test)

model = LogisticRegression(
    C=0.00938, class_weight='balanced',
    max_iter=2000, random_state=42)
model.fit(X_dev_sc, y_dev)

print(f"Model retrained. Test AUC check: ", end="")
from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test_sc)[:, 1]
print(f"{roc_auc_score(y_test, y_prob):.3f}")

# ── 2. CLEAN FEATURE NAMES ───────────────────────────────────────────
FEATURE_LABELS = {
    'agecalc_adm':            'Age (months)',
    'weight_kg_adm':          'Weight (kg)',
    'muac_mm_adm':            'MUAC (mm)',
    'hr_bpm_adm':             'Heart Rate (bpm)',
    'rr_brpm_app_adm':        'Respiratory Rate',
    'temp_c_adm':             'Temperature (°C)',
    'spo2site1_pc_oxi_adm':   'SpO₂ (%)',
    'badhealthduration_adm':  'Duration of Illness',
    'sysbp_mmhg_adm':         'Systolic BP (mmHg)',
    'respdistress_adm':       'Respiratory Distress',
    'caprefill_adm':          'Capillary Refill ≥3s',
    'priorweekabx_adm':       'Prior Week Antibiotics',
    'prioryearcough_adm':     'Prior Year Cough/Dyspnoea',
    'tbcontact_adm':          'TB Household Contact',
    'malariastatuspos_adm':   'Malaria RDT Positive',
    'prioryearwheeze_adm':    'Prior Year Wheeze',
    'diarrheaoften_adm':      'Frequent Diarrhea',
    'symptoms_adm___2':       'Cough < 14 days',
    'symptoms_adm___3':       'Cough > 14 days',
    'symptoms_adm___4':       'Diarrhea < 14 days',
    'symptoms_adm___6':       'Fever < 7 days',
    'symptoms_adm___7':       'Fever ≥ 7 days',
    'symptoms_adm___8':       'Vomiting Everything',
    'symptoms_adm___9':       'Abnormally Sleepy',
    'symptoms_adm___10':      'Bilateral Foot Swelling',
    'symptoms_adm___14':      'Seizures / Convulsions',
    'symptoms_adm___15':      'Coma',
    'comorbidity_adm___1':    'Asthma / Reactive Airway',
    'comorbidity_adm___3':    'Cardiac Disease',
    'comorbidity_adm___5':    'Sickle Cell Disease',
    'comorbidity_adm___6':    'Tuberculosis',
    'comorbidity_adm___11':   'No Comorbidity',
    'sex_adm':                'Sex (Male=1)',
    'feedingstatus_adm':      'Feeding Status',
    'hivstatus_adm':          'HIV Positive',
    'vaccpneumoc_adm':        'PCV Doses Received',
    'oxygenavail_adm':        'Oxygen Being Used',
    'tachypnea_who':          'WHO Tachypnoea',
    'severe_hypoxia':         'Severe Hypoxia (SpO₂<90%)',
    'sam_muac':               'Severe Acute Malnutrition',
    'mam_muac':               'Moderate Acute Malnutrition',
    'bcs_eye_num':            'BCS Eye Score',
    'bcs_motor_num':          'BCS Motor Score',
    'bcs_verbal_num':         'BCS Verbal Score',
    'bcs_total':              'BCS Total Score',
}
labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]

# ── 3. SHAP VALUES ───────────────────────────────────────────────────
print("Computing SHAP values (LinearExplainer)...")
explainer   = shap.LinearExplainer(model, X_dev_sc, feature_perturbation="interventional")
shap_test   = explainer.shap_values(X_test_sc)
shap_all    = explainer.shap_values(scaler.transform(X))

print(f"SHAP array shape: {shap_test.shape}")

# Mean absolute SHAP per feature (global importance)
mean_abs_shap = np.abs(shap_test).mean(axis=0)
importance_df = pd.DataFrame({
    'feature':       FEATURES,
    'label':         labels,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print("\nTop 15 features by mean |SHAP|:")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv('/home/claude/project/outputs/shap_importance.csv', index=False)

# ── 4. FIGURES ───────────────────────────────────────────────────────
TEAL   = '#1A7A5E'
AMBER  = '#D97706'
RED    = '#B91C1C'
BLUE   = '#1D4ED8'
LGRAY  = '#6B7280'

TOP_N = 20
top_idx    = importance_df.head(TOP_N)['feature'].map(
    lambda f: FEATURES.index(f)).values
top_labels = importance_df.head(TOP_N)['label'].values
top_shap   = shap_test[:, top_idx]
top_X      = X_test_sc[:, top_idx]

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.38)

# ── Plot 1: Global bar chart (mean |SHAP|) ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
top15 = importance_df.head(15)
colors_bar = [TEAL if i < 5 else (AMBER if i < 10 else LGRAY)
              for i in range(15)]
bars = ax1.barh(range(15), top15['mean_abs_shap'].values,
                color=colors_bar, alpha=0.88, edgecolor='white')
ax1.set_yticks(range(15))
ax1.set_yticklabels(top15['label'].values, fontsize=9.5)
ax1.invert_yaxis()
ax1.set_xlabel('Mean |SHAP Value|', fontsize=10)
ax1.set_title('Global Feature Importance\n(Mean Absolute SHAP — Test Set)',
              fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    ax1.text(bar.get_width() + 0.0005,
             bar.get_y() + bar.get_height()/2,
             f'{top15["mean_abs_shap"].values[i]:.4f}',
             va='center', fontsize=8, color=LGRAY)
ax1.spines[['top','right']].set_visible(False)

# ── Plot 2: Beeswarm / dot plot ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
cmap = LinearSegmentedColormap.from_list('rg', ['#1D4ED8', '#E5E7EB', '#B91C1C'])
for i in range(TOP_N - 1, -1, -1):
    sv   = top_shap[:, i]
    fv   = top_X[:, i]
    fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
    jitter = np.random.RandomState(i).uniform(-0.35, 0.35, len(sv))
    colors = cmap(fv_n)
    ax2.scatter(sv, np.full_like(sv, i) + jitter,
                c=colors, s=6, alpha=0.55, linewidths=0)

ax2.set_yticks(range(TOP_N))
ax2.set_yticklabels(top_labels, fontsize=8.5)
ax2.axvline(0, color='k', lw=0.8, alpha=0.6)
ax2.set_xlabel('SHAP Value (impact on log-odds of poor outcome)', fontsize=9.5)
ax2.set_title('SHAP Beeswarm Plot\n(Top 20 Features — Test Set)',
              fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2, axis='x')
ax2.spines[['top','right']].set_visible(False)
# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, shrink=0.5, pad=0.02)
cbar.set_label('Feature Value\n(low → high)', fontsize=8)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Low', 'Mid', 'High'])

# ── Plot 3: SHAP dependence — MUAC ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
muac_idx   = FEATURES.index('muac_mm_adm')
spo2_idx   = FEATURES.index('spo2site1_pc_oxi_adm')
muac_raw   = df.loc[X_test_df.index, 'muac_mm_adm'] if hasattr(X_test_df,'index') else X_test[:, muac_idx]

# Use original unscaled test values
_, X_test_orig, _, _ = train_test_split(X_df, y, test_size=0.20, random_state=42, stratify=y)
muac_vals  = X_test_orig['muac_mm_adm'].values
spo2_vals  = X_test_orig['spo2site1_pc_oxi_adm'].values
muac_shap  = shap_test[:, muac_idx]

sc3 = ax3.scatter(muac_vals, muac_shap,
                  c=spo2_vals, cmap='RdYlGn',
                  s=14, alpha=0.65, linewidths=0,
                  vmin=80, vmax=100)
ax3.axhline(0, color='k', lw=0.8, alpha=0.5)
ax3.axvline(115, color=RED,   lw=1.5, linestyle='--', alpha=0.7, label='SAM cutoff (115mm)')
ax3.axvline(125, color=AMBER, lw=1.5, linestyle='--', alpha=0.7, label='MAM cutoff (125mm)')
plt.colorbar(sc3, ax=ax3, label='SpO₂ (%)', shrink=0.8)
ax3.set_xlabel('MUAC (mm)', fontsize=10)
ax3.set_ylabel('SHAP Value', fontsize=10)
ax3.set_title('SHAP Dependence: MUAC\n(coloured by SpO₂)',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.spines[['top','right']].set_visible(False)

# ── Plot 4: SHAP dependence — SpO₂ ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
spo2_shap = shap_test[:, spo2_idx]
rd_idx    = FEATURES.index('respdistress_adm')
rd_vals   = X_test_orig['respdistress_adm'].values

colors_rd = [RED if v == 1 else TEAL for v in rd_vals]
ax4.scatter(spo2_vals, spo2_shap,
            c=colors_rd, s=14, alpha=0.60, linewidths=0)
ax4.axhline(0, color='k', lw=0.8, alpha=0.5)
ax4.axvline(90, color=RED, lw=1.5, linestyle='--', alpha=0.7,
            label='Severe hypoxia (90%)')
ax4.axvline(95, color=AMBER, lw=1.5, linestyle='--', alpha=0.7,
            label='Mild hypoxia (95%)')
from matplotlib.patches import Patch
legend_els = [Patch(color=RED, label='Resp Distress: Yes'),
              Patch(color=TEAL, label='Resp Distress: No')]
ax4.legend(handles=legend_els, fontsize=8, loc='upper right')
ax4.set_xlabel('SpO₂ (%)', fontsize=10)
ax4.set_ylabel('SHAP Value', fontsize=10)
ax4.set_title('SHAP Dependence: SpO₂\n(coloured by Respiratory Distress)',
              fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.spines[['top','right']].set_visible(False)

# ── Plot 5: Waterfall for a high-risk patient ────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
# Pick the highest predicted probability patient
high_risk_idx = np.argmax(y_prob)
sv_single     = shap_test[high_risk_idx, :]
base_val      = explainer.expected_value

# Keep top 12 by absolute value for readability
top12_idx     = np.argsort(np.abs(sv_single))[-12:][::-1]
sv_top12      = sv_single[top12_idx]
lab_top12     = [labels[i] for i in top12_idx]
colors_wf     = [RED if v > 0 else TEAL for v in sv_top12]

ax5.barh(range(12), sv_top12[::-1],
         color=colors_wf[::-1], alpha=0.85, edgecolor='white')
ax5.set_yticks(range(12))
ax5.set_yticklabels(lab_top12[::-1], fontsize=9)
ax5.axvline(0, color='k', lw=0.8)
ax5.set_xlabel('SHAP Value', fontsize=10)
prob_val = y_prob[high_risk_idx]
ax5.set_title(f'Waterfall: Highest-Risk Patient\n'
              f'Predicted Probability = {prob_val:.3f} | '
              f'True Outcome = {"Poor" if y_test[high_risk_idx]==1 else "Good"}',
              fontsize=10, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
ax5.spines[['top','right']].set_visible(False)

# ── Plot 6: Waterfall for a low-risk patient ─────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
good_outcome_idx = np.where(y_test == 0)[0]
low_risk_idx     = good_outcome_idx[np.argmin(y_prob[good_outcome_idx])]
sv_low           = shap_test[low_risk_idx, :]

top12_low        = np.argsort(np.abs(sv_low))[-12:][::-1]
sv_top12_low     = sv_low[top12_low]
lab_top12_low    = [labels[i] for i in top12_low]
colors_wf_low    = [RED if v > 0 else TEAL for v in sv_top12_low]

ax6.barh(range(12), sv_top12_low[::-1],
         color=colors_wf_low[::-1], alpha=0.85, edgecolor='white')
ax6.set_yticks(range(12))
ax6.set_yticklabels(lab_top12_low[::-1], fontsize=9)
ax6.axvline(0, color='k', lw=0.8)
ax6.set_xlabel('SHAP Value', fontsize=10)
prob_low = y_prob[low_risk_idx]
ax6.set_title(f'Waterfall: Lowest-Risk Patient\n'
              f'Predicted Probability = {prob_low:.3f} | '
              f'True Outcome = {"Poor" if y_test[low_risk_idx]==1 else "Good"}',
              fontsize=10, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')
ax6.spines[['top','right']].set_visible(False)

fig.suptitle(
    'SHAP Explainability Analysis — Logistic Regression (Best Model)\n'
    'Pediatric Sepsis Poor Outcome Prediction | Md Rafin Rahman, ideSHi',
    fontsize=13, fontweight='bold', y=1.005)

plt.savefig('/home/claude/project/figures/shap_analysis.png',
            dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print("SHAP figure saved.")

# ── 5. LOGISTIC REGRESSION COEFFICIENTS TABLE ────────────────────────
coef_df = pd.DataFrame({
    'feature':    FEATURES,
    'label':      labels,
    'coef':       model.coef_[0],
    'abs_coef':   np.abs(model.coef_[0]),
    'mean_abs_shap': mean_abs_shap,
    'direction':  ['Risk factor' if c > 0 else 'Protective' for c in model.coef_[0]]
}).sort_values('abs_coef', ascending=False).reset_index(drop=True)

coef_df.to_csv('/home/claude/project/outputs/lr_coefficients.csv', index=False)

print("\nTop 20 LR Coefficients (standardised):")
print(coef_df[['label','coef','direction','mean_abs_shap']].head(20).to_string(index=False))

# ── 6. SUMMARY STATS FOR TOP FEATURES ────────────────────────────────
print("\n=== FEATURE SUMMARY BY OUTCOME ===")
top10_feats = importance_df.head(10)['feature'].tolist()
for feat in top10_feats:
    label = FEATURE_LABELS.get(feat, feat)
    good  = df.loc[df['poor_outcome']==0, feat]
    poor  = df.loc[df['poor_outcome']==1, feat]
    if df[feat].nunique() <= 3:
        good_rate = good.mean()
        poor_rate = poor.mean()
        print(f"  {label:<35}  Good={good_rate:.3f}  Poor={poor_rate:.3f}  "
              f"Diff={poor_rate-good_rate:+.3f}")
    else:
        print(f"  {label:<35}  Good={good.mean():.2f}±{good.std():.2f}  "
              f"Poor={poor.mean():.2f}±{poor.std():.2f}  "
              f"Diff={poor.mean()-good.mean():+.2f}")

print("\n✓ Step 04 complete.")
