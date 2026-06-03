"""
Step 03: Hyperparameter Tuning with Optuna
Project: Predicting Poor Clinical Outcome in Pediatric Sepsis
Author:  Md Rafin Rahman, ideSHi
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              precision_score, average_precision_score,
                              brier_score_loss, confusion_matrix,
                              roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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

from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_TRIALS = 80

print(f"Tuning on development set: {X_dev.shape[0]} records")
print(f"Holdout test set:          {X_test.shape[0]} records")
print(f"Optuna trials per model:   {N_TRIALS}")

# ── 2. OBJECTIVE FUNCTIONS ────────────────────────────────────────────

def objective_lr(trial):
    C     = trial.suggest_float('C', 1e-3, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
    penalty = 'l2'
    if solver == 'saga':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C, solver=solver, penalty=penalty,
            class_weight='balanced', max_iter=3000, random_state=42))
    ])
    scores = cross_val_score(model, X_dev, y_dev, cv=CV,
                             scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def objective_rf(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 800),
        'max_depth':       trial.suggest_int('max_depth', 3, 12),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf', 5, 30),
        'min_samples_split':trial.suggest_int('min_samples_split', 5, 30),
        'max_features':    trial.suggest_categorical('max_features',
                                                     ['sqrt', 'log2', 0.5, 0.7]),
        'class_weight':    'balanced',
        'random_state':    42,
        'n_jobs':          -1,
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_dev, y_dev, cv=CV,
                             scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def objective_xgb(trial):
    scale_pos = (1 - y_dev).sum() / y_dev.sum()
    params = {
        'n_estimators':        trial.suggest_int('n_estimators', 200, 800),
        'max_depth':           trial.suggest_int('max_depth', 3, 8),
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':           trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':    trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':    trial.suggest_int('min_child_weight', 1, 10),
        'gamma':               trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha':           trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda':          trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'scale_pos_weight':    scale_pos,
        'eval_metric':         'logloss',
        'random_state':        42,
        'verbosity':           0,
        'use_label_encoder':   False,
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_dev, y_dev, cv=CV,
                             scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def objective_lgb(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'num_leaves':       trial.suggest_int('num_leaves', 15, 63),
        'class_weight':     'balanced',
        'random_state':     42,
        'verbose':          -1,
        'n_jobs':           -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_dev, y_dev, cv=CV,
                             scoring='roc_auc', n_jobs=-1)
    return scores.mean()

# ── 3. RUN TUNING ─────────────────────────────────────────────────────
objectives = {
    'Logistic Regression': objective_lr,
    'Random Forest':       objective_rf,
    'XGBoost':             objective_xgb,
    'LightGBM':            objective_lgb,
}

best_params = {}
best_cv_auc = {}

for name, obj in objectives.items():
    print(f"\nTuning {name} ({N_TRIALS} trials)...")
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    best_params[name] = study.best_params
    best_cv_auc[name] = study.best_value
    print(f"  Best CV AUC-ROC: {study.best_value:.4f}")
    print(f"  Best params:     {study.best_params}")

# ── 4. TRAIN TUNED MODELS ─────────────────────────────────────────────
print("\n=== TRAINING TUNED MODELS ON FULL DEVELOPMENT SET ===")

def build_tuned_model(name, params):
    if name == 'Logistic Regression':
        p = dict(params)
        solver  = p.pop('solver')
        penalty = p.pop('penalty', 'l2')
        C       = p.pop('C')
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=C, solver=solver, penalty=penalty,
                class_weight='balanced', max_iter=3000, random_state=42))
        ])
    elif name == 'Random Forest':
        return RandomForestClassifier(
            **params, class_weight='balanced',
            random_state=42, n_jobs=-1)
    elif name == 'XGBoost':
        return xgb.XGBClassifier(
            **params,
            scale_pos_weight=(1-y_dev).sum()/y_dev.sum(),
            eval_metric='logloss', random_state=42,
            verbosity=0, use_label_encoder=False)
    elif name == 'LightGBM':
        return lgb.LGBMClassifier(
            **params, class_weight='balanced',
            random_state=42, verbose=-1, n_jobs=-1)

tuned_models = {}
for name, params in best_params.items():
    model = build_tuned_model(name, params)
    model.fit(X_dev, y_dev)
    tuned_models[name] = model
    print(f"  {name}: trained")

# ── 5. EVALUATE ON HOLDOUT ────────────────────────────────────────────
print("\n=== TUNED MODEL PERFORMANCE — HOLDOUT TEST SET ===")

COLORS = {
    'Logistic Regression': '#1A7A5E',
    'Random Forest':       '#D97706',
    'XGBoost':             '#1D4ED8',
    'LightGBM':            '#B91C1C',
}

tuned_results = {}
for name, model in tuned_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Optimal sensitivity threshold (Youden J)
    fpr_c, tpr_c, thresh_c = roc_curve(y_test, y_prob)
    youden_j    = tpr_c - fpr_c
    opt_thresh  = thresh_c[np.argmax(youden_j)]
    y_pred_opt  = (y_prob >= opt_thresh).astype(int)
    cm_opt      = confusion_matrix(y_test, y_pred_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()

    tuned_results[name] = {
        'AUC-ROC':          roc_auc_score(y_test, y_prob),
        'AUC-PR':           average_precision_score(y_test, y_prob),
        'F1':               f1_score(y_test, y_pred),
        'Sensitivity':      recall_score(y_test, y_pred),
        'Specificity':      specificity,
        'Precision':        precision_score(y_test, y_pred),
        'Brier':            brier_score_loss(y_test, y_prob),
        'Opt_threshold':    float(opt_thresh),
        'Opt_Sensitivity':  tp_o/(tp_o+fn_o),
        'Opt_Specificity':  tn_o/(tn_o+fp_o),
        'Opt_F1':           f1_score(y_test, y_pred_opt),
        'CV_AUC':           best_cv_auc[name],
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'y_prob': y_prob,
    }
    print(f"\n{name}  (CV AUC={best_cv_auc[name]:.4f}):")
    r = tuned_results[name]
    print(f"  Default threshold (0.50):")
    print(f"    AUC-ROC={r['AUC-ROC']:.3f}  AUC-PR={r['AUC-PR']:.3f}  "
          f"F1={r['F1']:.3f}  Sens={r['Sensitivity']:.3f}  "
          f"Spec={r['Specificity']:.3f}  Brier={r['Brier']:.3f}")
    print(f"    Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Youden J threshold ({opt_thresh:.2f}):")
    print(f"    Sens={r['Opt_Sensitivity']:.3f}  "
          f"Spec={r['Opt_Specificity']:.3f}  F1={r['Opt_F1']:.3f}")

# ── 6. IDENTIFY BEST MODEL ────────────────────────────────────────────
best_name = max(tuned_results, key=lambda k: tuned_results[k]['AUC-ROC'])
print(f"\nBest tuned model: {best_name} "
      f"(AUC-ROC = {tuned_results[best_name]['AUC-ROC']:.3f})")

# ── 7. SAVE ───────────────────────────────────────────────────────────
save_tuned = {}
for name, res in tuned_results.items():
    save_tuned[name] = {k: round(float(v), 4)
                        for k, v in res.items() if k != 'y_prob'}

with open('/home/claude/project/outputs/tuned_results.json', 'w') as f:
    json.dump(save_tuned, f, indent=2)
with open('/home/claude/project/outputs/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)
with open('/home/claude/project/outputs/best_model_name.txt', 'w') as f:
    f.write(best_name)

# ── 8. FIGURES ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# Plot 1: ROC curves
ax1 = fig.add_subplot(gs[0, 0])
for name, res in tuned_results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax1.plot(fpr, tpr, color=COLORS[name], lw=2,
             label=f"{name} ({res['AUC-ROC']:.3f})")
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random (0.500)')
ax1.set_xlabel('1 - Specificity', fontsize=10)
ax1.set_ylabel('Sensitivity', fontsize=10)
ax1.set_title('ROC Curves — Tuned Models', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)

# Plot 2: Precision-Recall
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(y_test.mean(), color='k', linestyle='--', lw=1,
            alpha=0.5, label=f'Baseline ({y_test.mean():.3f})')
for name, res in tuned_results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax2.plot(rec, prec, color=COLORS[name], lw=2,
             label=f"{name} (PR={res['AUC-PR']:.3f})")
ax2.set_xlabel('Recall', fontsize=10)
ax2.set_ylabel('Precision', fontsize=10)
ax2.set_title('Precision-Recall Curves — Tuned Models', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Before vs After Tuning AUC comparison
ax3 = fig.add_subplot(gs[0, 2])
with open('/home/claude/project/outputs/test_results.json') as f:
    baseline_results = json.load(f)
model_names = list(tuned_results.keys())
before_auc  = [baseline_results[m]['AUC-ROC'] for m in model_names]
after_auc   = [tuned_results[m]['AUC-ROC']    for m in model_names]
x = np.arange(len(model_names))
w = 0.35
bars1 = ax3.bar(x - w/2, before_auc, w, label='Before tuning',
                color='#9CA3AF', alpha=0.8)
bars2 = ax3.bar(x + w/2, after_auc,  w, label='After tuning',
                color=[COLORS[m] for m in model_names], alpha=0.9)
ax3.set_xticks(x)
ax3.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8.5)
ax3.set_ylabel('AUC-ROC', fontsize=10)
ax3.set_title('AUC-ROC: Before vs After Tuning', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_ylim([0.55, 0.80])
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# Plot 4: Calibration curves
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Perfect')
for name, res in tuned_results.items():
    prob_true, prob_pred = calibration_curve(
        y_test, res['y_prob'], n_bins=10, strategy='uniform')
    ax4.plot(prob_pred, prob_true, 'o-', color=COLORS[name],
             lw=2, ms=5, label=f"{name} (Brier={res['Brier']:.3f})")
ax4.set_xlabel('Mean Predicted Probability', fontsize=10)
ax4.set_ylabel('Fraction of Positives', fontsize=10)
ax4.set_title('Calibration Curves — Tuned Models', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Performance heatmap
ax5 = fig.add_subplot(gs[1, 1])
metric_names = ['AUC-ROC','AUC-PR','F1','Sensitivity','Specificity','Precision']
heatmap_data = np.array([[tuned_results[m][k] for k in metric_names]
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
ax5.set_title('Performance Heatmap — Tuned Models', fontsize=11, fontweight='bold')

# Plot 6: Threshold analysis for best model
ax6 = fig.add_subplot(gs[1, 2])
best_prob  = tuned_results[best_name]['y_prob']
thresholds = np.arange(0.1, 0.91, 0.01)
sens_l, spec_l, f1_l, ppv_l = [], [], [], []
for t in thresholds:
    pred = (best_prob >= t).astype(int)
    cm   = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    sens_l.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    spec_l.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    f1_l.append(f1_score(y_test, pred, zero_division=0))
    ppv_l.append(tp/(tp+fp) if (tp+fp)>0 else 0)

ax6.plot(thresholds, sens_l, color='#1A7A5E', lw=2, label='Sensitivity')
ax6.plot(thresholds, spec_l, color='#1D4ED8', lw=2, label='Specificity')
ax6.plot(thresholds, f1_l,   color='#D97706', lw=2, label='F1 Score')
ax6.plot(thresholds, ppv_l,  color='#7C3AED', lw=2, label='Precision (PPV)')
opt_t = tuned_results[best_name]['Opt_threshold']
ax6.axvline(opt_t,  color='#B91C1C', linestyle=':', lw=2,
            label=f'Youden J ({opt_t:.2f})')
ax6.axvline(0.5,    color='k',       linestyle='--',lw=1, alpha=0.5,
            label='Default (0.50)')
ax6.set_xlabel('Decision Threshold', fontsize=10)
ax6.set_ylabel('Score', fontsize=10)
ax6.set_title(f'Threshold Analysis — {best_name}', fontsize=11, fontweight='bold')
ax6.legend(fontsize=7.5)
ax6.grid(True, alpha=0.3)
ax6.set_xlim([0.1,0.9]); ax6.set_ylim([0,1.02])

fig.suptitle(
    'Tuned Model Evaluation — Pediatric Sepsis Poor Outcome Prediction\n'
    'Point-of-Care Feature Constraint | Uganda LMIC Cohort | Md Rafin Rahman, ideSHi',
    fontsize=12, fontweight='bold', y=1.01)

out_path = '/home/claude/project/figures/tuned_model_evaluation.png'
plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure saved: tuned_model_evaluation.png")

# ── 9. FINAL SUMMARY ──────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL TUNED PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Model':<22} {'CV AUC':>8} {'Test AUC':>9} {'AUC-PR':>8} "
      f"{'F1':>7} {'Sens':>7} {'Spec':>7} {'Brier':>7}")
print("-"*80)
for name in model_names:
    r = tuned_results[name]
    marker = " *" if name == best_name else ""
    print(f"{name+marker:<22} {r['CV_AUC']:>8.3f} {r['AUC-ROC']:>9.3f} "
          f"{r['AUC-PR']:>8.3f} {r['F1']:>7.3f} {r['Sensitivity']:>7.3f} "
          f"{r['Specificity']:>7.3f} {r['Brier']:>7.3f}")
print("="*80)
print(f"\nYouden J threshold performance for {best_name}:")
r = tuned_results[best_name]
print(f"  Threshold:   {r['Opt_threshold']:.2f}")
print(f"  Sensitivity: {r['Opt_Sensitivity']:.3f}")
print(f"  Specificity: {r['Opt_Specificity']:.3f}")
print(f"  F1:          {r['Opt_F1']:.3f}")
print(f"\n✓ Step 03 complete. Proceed to SHAP analysis on: {best_name}")
