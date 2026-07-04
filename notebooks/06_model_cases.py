"""
Musical DNA — Week 6: Copyright Case Outcome Predictor
========================================================
Trains logistic regression models predicting whether a court found
infringement, comparing three feature sets to answer the blueprint's core
question: does musical similarity predict outcomes, or do non-musical
factors (fame, commercial success, forum) predict them just as well?

Evaluation: stratified 5-fold CV + leave-one-out (33 cases is too small for
a trustworthy single train/test split).

Usage:
    python notebooks/06_model_cases.py

Reads:  data/copyright_cases/cases.csv (33 fully-scored rows)
Saves:  models/case_predictor.pkl
        notebooks/plots/07_case_model_coefficients.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.case_model import (
    FEATURE_SETS, load_case_features, build_pipeline,
    evaluate_model, get_coefficients, loo_predictions, save_model,
)

PLOT_DIR = 'notebooks/plots'
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_feature_sets(df):
    print(f"\n── FEATURE SET COMPARISON ───────────────────────────────")
    print(f"  {'Feature set':<14} {'Accuracy (5-fold)':>18} {'AUC-ROC':>16} {'LOO Acc':>9}")
    print('  ' + '-' * 62)

    results = {}
    for name, features in FEATURE_SETS.items():
        X, y, _ = load_case_features(feature_set=name)
        pipeline = build_pipeline()
        result = evaluate_model(pipeline, X, y)
        results[name] = result

        acc_mean, acc_std = result['kfold']['accuracy']
        auc_mean, auc_std = result['kfold']['roc_auc']
        print(f"  {name:<14} {acc_mean:>8.1%} ± {acc_std:<6.1%} "
              f"{auc_mean:>8.2f} ± {auc_std:<5.2f} {result['loo_accuracy']:>8.1%}")

    return results


def print_precision_recall(results):
    print(f"\n  Precision / Recall (5-fold mean):")
    print(f"  {'Feature set':<14} {'Precision':>10} {'Recall':>8}")
    print('  ' + '-' * 36)
    for name, result in results.items():
        p_mean, _ = result['kfold']['precision']
        r_mean, _ = result['kfold']['recall']
        print(f"  {name:<14} {p_mean:>10.1%} {r_mean:>8.1%}")


# =============================================================================
# COEFFICIENT INTERPRETATION
# =============================================================================

def fit_full_model_and_interpret():
    X, y, df = load_case_features(feature_set='full')
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    coef_df = get_coefficients(pipeline, FEATURE_SETS['full'])
    print(f"\n── FULL MODEL — STANDARDIZED COEFFICIENTS ───────────────")
    print(f"  (fit on all {len(y)} scored cases; odds ratio > 1 → pushes toward 'infringement found')")
    print(f"  {'Feature':<32} {'Coefficient':>12} {'Odds Ratio':>11}")
    print('  ' + '-' * 57)
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:<32} {row['coefficient']:>12.3f} {row['odds_ratio']:>11.2f}")

    return pipeline, coef_df, X, y, df


def plot_coefficients(coef_df):
    coef_df = coef_df.iloc[::-1]  # largest at top of horizontal bar chart
    colors = ['#d62728' if c < 0 else '#2ca02c' for c in coef_df['coefficient']]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='white')
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Standardized Coefficient (+ → toward infringement found)', fontsize=11)
    ax.set_title('Copyright Case Outcome — Logistic Regression Coefficients',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = f'{PLOT_DIR}/07_case_model_coefficients.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


# =============================================================================
# OUTLIER CASES
# =============================================================================

def print_outlier_cases(df, X, y, top_n=5):
    """Cases where out-of-fold predicted probability most disagrees with the
    actual ruling — the "surprising findings" the blueprint milestone asks for.
    """
    pipeline = build_pipeline()
    probs = loo_predictions(pipeline, X, y)
    df = df.copy()
    df['predicted_prob'] = probs
    df['residual'] = df['predicted_prob'] - df['outcome']

    print(f"\n── OUTLIER CASES (leave-one-out predictions) ────────────")
    print(f"  Cases where the model's prediction most disagrees with the court's ruling:")
    print(f"  {'Case':<45} {'Ruling':>8} {'Pred. P(infringe)':>18}")
    print('  ' + '-' * 73)

    outliers = df.reindex(df['residual'].abs().sort_values(ascending=False).index).head(top_n)
    for _, row in outliers.iterrows():
        ruling = 'Infringe' if row['outcome'] == 1 else 'No infr.'
        print(f"  {row['case_name']:<45} {ruling:>8} {row['predicted_prob']:>18.2f}")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 60)
    print('  MUSICAL DNA — Week 6: Copyright Case Outcome Predictor')
    print('=' * 60)

    _, y_full, df = load_case_features(feature_set='full')
    n_infringe = int(y_full.sum())
    print(f"\n  Dataset: {len(y_full)} scored cases "
          f"({n_infringe} infringement / {len(y_full) - n_infringe} no infringement)")
    print(f"  Note: with only {len(y_full)} cases, cross-validation is used throughout —")
    print(f"        a single train/test split would be too noisy to trust.")

    results = compare_feature_sets(df)
    print_precision_recall(results)

    musical_auc = results['musical']['kfold']['roc_auc'][0]
    non_musical_auc = results['non_musical']['kfold']['roc_auc'][0]
    full_auc = results['full']['kfold']['roc_auc'][0]

    print(f"\n── RESEARCH QUESTION ─────────────────────────────────────")
    print(f"  Musical similarity alone:      AUC = {musical_auc:.2f}")
    print(f"  Non-musical factors alone:     AUC = {non_musical_auc:.2f}")
    print(f"  Combined:                      AUC = {full_auc:.2f}")
    better = 'non-musical factors' if non_musical_auc > musical_auc else 'musical similarity'
    print(f"\n  → {better} is the stronger predictor on its own "
          f"({max(musical_auc, non_musical_auc):.2f} vs {min(musical_auc, non_musical_auc):.2f} AUC).")

    pipeline, coef_df, X, y, df = fit_full_model_and_interpret()
    plot_coefficients(coef_df)

    df_with_preds = print_outlier_cases(df, X, y)

    print(f"\n── SAVING MODEL ─────────────────────────────────────────")
    save_model(pipeline)
    print(f"  Saved: models/case_predictor.pkl (full feature set, fit on all {len(y)} cases)")

    print(f"\n{'=' * 60}")
    print(f"  WEEK 6 MILESTONE")
    print(f"{'=' * 60}")
    print(f"\n  {len(y)} cases scored. Logistic regression trained and cross-validated.")
    print(f"  Strongest single predictor: {coef_df.iloc[0]['feature']} "
          f"(coefficient {coef_df.iloc[0]['coefficient']:+.2f})")
    print(f"\n  Next step: Week 7 — deeper outlier analysis + synthesis with Component A")


if __name__ == '__main__':
    main()
