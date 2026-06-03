"""
Step 02: Model Training and Evaluation
Project: Predicting Poor Clinical Outcome in Pediatric Sepsis
         Using Point-of-Care Features in a Low-Resource Setting
Author:  Md Rafin Rahman, ideSHi
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              precision_score, average_precision_score,
                              brier_score_loss, confusion_matrix,
                              roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. LOAD ───────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/project/data/processed.csv')
with open('/home/claude/project/data/feature_list.json') as f:
    FEATURES = json.load(f)

X = df[FEATURES].values
y = df['poor_outcome'].values
print(f"Dataset: {X.shape[0]} records, {X.shape[1]} features")
print(f"Outcome: {y.sum()} positive ({100*y.mean():.1f}%), {(1-y).sum()} negative")

# ── 2. STRATIFIED HOLDOUT ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nDevelopment set: {X_dev.shape[0]} records "
      f"({100*y_dev.mean():.1f}% positive)")
print(f"Holdout test:    {X_test.shape[0]} records "
      f"({100*y_test.mean():.1f}% positive)")

# ── 3. CROSS-VALIDATION SETUP ─────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 4. MODEL DEFINITIONS ──────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=2000, class_weight='balanced',
            C=0.1, solver='lbfgs', random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1-y_dev).sum()/y_dev.sum(),
        eval_metric='logloss', random_state=42,
        verbosity=0, use_label_encoder=False),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42,
        verbose=-1, n_jobs=-1),
}

# ── 5. CROSS-VALIDATION ON DEVELOPMENT SET ───────────────────────────
print("\n=== 5-FOLD CROSS-VALIDATION (development set) ===")
cv_results = {}

scoring = {
    'roc_auc':          'roc_auc',
    'average_precision':'average_precision',
    'f1':               'f1',
    'recall':           'recall',
    'precision':        'precision',
}

for name, model in models.items():
    cv = cross_validate(model, X_dev, y_dev, cv=CV,
                        scoring=scoring, return_train_score=False, n_jobs=-1)
    cv_results[name] = {
        'AUC-ROC':    (cv['test_roc_auc'].mean(),    cv['test_roc_auc'].std()),
        'AUC-PR':     (cv['test_average_precision'].mean(),
                       cv['test_average_precision'].std()),
        'F1':         (cv['test_f1'].mean(),          cv['test_f1'].std()),
        'Sensitivity':(cv['test_recall'].mean(),       cv['test_recall'].std()),
        'Precision':  (cv['test_precision'].mean(),    cv['test_precision'].std()),
    }
    print(f"\n{name}:")
    for metric, (mean, std) in cv_results[name].items():
        print(f"  {metric:<15} {mean:.3f} ± {std:.3f}")

# ── 6. TRAIN FINAL MODELS ON FULL DEVELOPMENT SET ────────────────────
print("\n=== TRAINING FINAL MODELS ON FULL DEVELOPMENT SET ===")
trained = {}
for name, model in models.items():
    model.fit(X_dev, y_dev)
    trained[name] = model
    print(f"  {name}: trained")

# ── 7. HOLDOUT TEST SET EVALUATION ───────────────────────────────────
print("\n=== HOLDOUT TEST SET EVALUATION ===")
test_results = {}
for name, model in trained.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    test_results[name] = {
        'AUC-ROC':     roc_auc_score(y_test, y_prob),
        'AUC-PR':      average_precision_score(y_test, y_prob),
        'F1':          f1_score(y_test, y_pred),
        'Sensitivity': recall_score(y_test, y_pred),
        'Specificity': specificity,
        'Precision':   precision_score(y_test, y_pred),
        'Brier':       brier_score_loss(y_test, y_prob),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'y_prob': y_prob,
    }
    print(f"\n{name}:")
    for m in ['AUC-ROC','AUC-PR','F1','Sensitivity','Specificity','Precision','Brier']:
        print(f"  {m:<15} {test_results[name][m]:.3f}")
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

# ── 8. SELECT BEST MODEL ──────────────────────────────────────────────
best_name = max(test_results, key=lambda k: test_results[k]['AUC-ROC'])
print(f"\nBest model by AUC-ROC: {best_name} "
      f"(AUC = {test_results[best_name]['AUC-ROC']:.3f})")

# ── 9. SAVE RESULTS ───────────────────────────────────────────────────
# Save numeric results (exclude y_prob array)
save_results = {}
for name, res in test_results.items():
    save_results[name] = {k: round(float(v), 4)
                          for k, v in res.items()
                          if k != 'y_prob'}

save_cv = {}
for name, res in cv_results.items():
    save_cv[name] = {k: {'mean': round(float(v[0]),4),
                         'std':  round(float(v[1]),4)}
                     for k, v in res.items()}

with open('/home/claude/project/outputs/test_results.json', 'w') as f:
    json.dump(save_results, f, indent=2)
with open('/home/claude/project/outputs/cv_results.json', 'w') as f:
    json.dump(save_cv, f, indent=2)
with open('/home/claude/project/outputs/best_model_name.txt', 'w') as f:
    f.write(best_name)

# ── 10. FIGURES ───────────────────────────────────────────────────────
COLORS = {
    'Logistic Regression': '#1A7A5E',
    'Random Forest':       '#D97706',
    'XGBoost':             '#1D4ED8',
    'LightGBM':            '#B91C1C',
}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# --- Plot 1: ROC Curves ---
ax1 = fig.add_subplot(gs[0, 0])
for name, res in test_results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax1.plot(fpr, tpr, color=COLORS[name], lw=2,
             label=f"{name} (AUC={res['AUC-ROC']:.3f})")
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random (AUC=0.500)')
ax1.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=10)
ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=10)
ax1.set_title('ROC Curves — Holdout Test Set', fontsize=11, fontweight='bold')
ax1.legend(fontsize=7.5, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0,1]); ax1.set_ylim([0,1.02])

# --- Plot 2: Precision-Recall Curves ---
ax2 = fig.add_subplot(gs[0, 1])
baseline_pr = y_test.mean()
ax2.axhline(baseline_pr, color='k', linestyle='--', lw=1,
            alpha=0.5, label=f'Baseline (PR={baseline_pr:.3f})')
for name, res in test_results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax2.plot(rec, prec, color=COLORS[name], lw=2,
             label=f"{name} (AUC-PR={res['AUC-PR']:.3f})")
ax2.set_xlabel('Recall (Sensitivity)', fontsize=10)
ax2.set_ylabel('Precision (PPV)', fontsize=10)
ax2.set_title('Precision-Recall Curves — Holdout Test Set', fontsize=11, fontweight='bold')
ax2.legend(fontsize=7.5, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])

# --- Plot 3: CV Performance Bar Chart ---
ax3 = fig.add_subplot(gs[0, 2])
metrics_to_plot = ['AUC-ROC', 'AUC-PR', 'F1', 'Sensitivity', 'Precision']
x = np.arange(len(metrics_to_plot))
width = 0.2
for i, (name, res) in enumerate(cv_results.items()):
    means = [res[m][0] for m in metrics_to_plot]
    stds  = [res[m][1] for m in metrics_to_plot]
    ax3.bar(x + i*width, means, width, yerr=stds,
            label=name, color=COLORS[name], alpha=0.85,
            error_kw={'elinewidth':1.2,'capsize':3})
ax3.set_xticks(x + width*1.5)
ax3.set_xticklabels(metrics_to_plot, fontsize=8.5, rotation=15)
ax3.set_ylabel('Score', fontsize=10)
ax3.set_title('5-Fold CV Performance (development set)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=7.5)
ax3.set_ylim([0, 1.05])
ax3.grid(True, alpha=0.3, axis='y')

# --- Plot 4: Calibration Curves ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Perfect calibration')
for name, res in test_results.items():
    prob_true, prob_pred = calibration_curve(
        y_test, res['y_prob'], n_bins=10, strategy='uniform')
    ax4.plot(prob_pred, prob_true, 'o-', color=COLORS[name],
             lw=2, ms=5, label=f"{name} (Brier={res['Brier']:.3f})")
ax4.set_xlabel('Mean Predicted Probability', fontsize=10)
ax4.set_ylabel('Fraction of Positives', fontsize=10)
ax4.set_title('Calibration Curves — Holdout Test Set', fontsize=11, fontweight='bold')
ax4.legend(fontsize=7.5)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0,1]); ax4.set_ylim([0,1])

# --- Plot 5: Metric Summary Heatmap ---
ax5 = fig.add_subplot(gs[1, 1])
metric_names = ['AUC-ROC','AUC-PR','F1','Sensitivity','Specificity','Precision']
model_names  = list(test_results.keys())
heatmap_data = np.array([[test_results[m][k] for k in metric_names]
                          for m in model_names])
im = ax5.imshow(heatmap_data, cmap='YlGn', vmin=0.4, vmax=0.9, aspect='auto')
ax5.set_xticks(range(len(metric_names)))
ax5.set_yticks(range(len(model_names)))
ax5.set_xticklabels(metric_names, fontsize=8.5, rotation=30, ha='right')
ax5.set_yticklabels(model_names, fontsize=9)
for i in range(len(model_names)):
    for j in range(len(metric_names)):
        ax5.text(j, i, f"{heatmap_data[i,j]:.3f}",
                 ha='center', va='center', fontsize=8.5,
                 color='black' if heatmap_data[i,j] < 0.75 else 'white',
                 fontweight='bold')
plt.colorbar(im, ax=ax5, shrink=0.8)
ax5.set_title('Performance Summary — Holdout Test Set', fontsize=11, fontweight='bold')

# --- Plot 6: Threshold Analysis (best model) ---
ax6 = fig.add_subplot(gs[1, 2])
best_prob = test_results[best_name]['y_prob']
thresholds = np.arange(0.1, 0.91, 0.01)
sens_list, spec_list, f1_list = [], [], []
for t in thresholds:
    pred = (best_prob >= t).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    sens_list.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    spec_list.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    f1_list.append(f1_score(y_test, pred, zero_division=0))

ax6.plot(thresholds, sens_list, color='#1A7A5E', lw=2, label='Sensitivity')
ax6.plot(thresholds, spec_list, color='#1D4ED8', lw=2, label='Specificity')
ax6.plot(thresholds, f1_list,   color='#D97706', lw=2, label='F1 Score')
ax6.axvline(0.5, color='k', linestyle='--', lw=1, alpha=0.5, label='Default threshold')
# Mark optimal F1 threshold
opt_idx = np.argmax(f1_list)
ax6.axvline(thresholds[opt_idx], color='#B91C1C', linestyle=':', lw=1.5,
            label=f'Optimal F1 threshold ({thresholds[opt_idx]:.2f})')
ax6.set_xlabel('Decision Threshold', fontsize=10)
ax6.set_ylabel('Score', fontsize=10)
ax6.set_title(f'Threshold Analysis — {best_name}', fontsize=11, fontweight='bold')
ax6.legend(fontsize=7.5)
ax6.grid(True, alpha=0.3)
ax6.set_xlim([0.1, 0.9]); ax6.set_ylim([0, 1.02])

fig.suptitle(
    'Predicting Poor Clinical Outcome in Pediatric Sepsis\n'
    'Point-of-Care Feature Constraint | Uganda LMIC Cohort | Md Rafin Rahman, ideSHi',
    fontsize=13, fontweight='bold', y=1.01)

plt.savefig('/home/claude/project/figures/model_evaluation.png',
            dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print("\nFigure saved: model_evaluation.png")

# ── 11. PRINT FINAL SUMMARY TABLE ────────────────────────────────────
print("\n" + "="*72)
print("FINAL PERFORMANCE SUMMARY — HOLDOUT TEST SET")
print("="*72)
header = f"{'Model':<22} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>7} {'Sens':>7} {'Spec':>7} {'Brier':>7}"
print(header)
print("-"*72)
for name, res in test_results.items():
    marker = " *" if name == best_name else ""
    print(f"{name+marker:<22} {res['AUC-ROC']:>8.3f} {res['AUC-PR']:>8.3f} "
          f"{res['F1']:>7.3f} {res['Sensitivity']:>7.3f} "
          f"{res['Specificity']:>7.3f} {res['Brier']:>7.3f}")
print("="*72)
print(f"* Best model by AUC-ROC")
print("\n✓ Step 02 complete.")
