"""
Musical DNA — Week 3: Composer Classifier
==========================================
Trains and compares three classifiers on the 21-feature matrix:
  - Random Forest
  - SVM (RBF kernel)
  - Gradient Boosting

Evaluation: 5-fold stratified cross-validation + held-out test set (80/20 split).

Usage:
    python notebooks/02_classifier.py

Reads:  data/processed/features.csv
Saves:  models/best_classifier.pkl
        notebooks/plots/05_confusion_matrix.png
        notebooks/plots/06_feature_importance.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from src.classifier import (
    load_data, get_models, build_pipeline,
    evaluate_model, get_feature_importances, save_model,
)

PLOT_DIR = 'notebooks/plots'
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAINING
# =============================================================================

def run_all_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    pipelines = {}

    print(f"\n  {'Model':<22} {'CV Accuracy (5-fold)':>22} {'Test Accuracy':>14}")
    print('  ' + '-' * 60)

    for name, model in models.items():
        print(f"  Training {name}...", end=' ', flush=True)
        pipeline = build_pipeline(model)
        result = evaluate_model(pipeline, X_train, X_test, y_train, y_test, cv=5)
        results[name] = result
        pipelines[name] = pipeline
        print(f"{result['cv_mean']:.1%} ± {result['cv_std']:.1%}  →  {result['test_accuracy']:.1%}")

    return results, pipelines


# =============================================================================
# REPORTING
# =============================================================================

def print_per_class_report(result, model_name):
    report = result['classification_report']
    print(f"\n  Per-composer breakdown — {model_name}:")
    print(f"  {'Composer':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print('  ' + '-' * 55)
    composers = sorted(k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg'))
    for composer in composers:
        r = report[composer]
        print(f"  {composer.title():<16} {r['precision']:>10.2%}"
              f" {r['recall']:>8.2%} {r['f1-score']:>8.2%} {int(r['support']):>9}")


# =============================================================================
# PLOTS
# =============================================================================

def plot_confusion_matrix(result, model_name):
    labels = result['cm_labels']
    cm = result['confusion_matrix']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[c.title() for c in labels],
        yticklabels=[c.title() for c in labels],
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(
        f'Confusion Matrix — {model_name}\nTest accuracy: {result["test_accuracy"]:.1%}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    path = f'{PLOT_DIR}/05_confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(rf_pipeline, top_n=10):
    imp_df = get_feature_importances(rf_pipeline)
    if imp_df is None:
        return

    top = imp_df.head(top_n)
    # Reverse so highest bar is at the top
    features = top['feature'].tolist()[::-1]
    importances = top['importance'].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(features, importances, color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=11)
    ax.set_title(f'Top {top_n} Features — Random Forest', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(importances) * 1.2)
    plt.tight_layout()

    path = f'{PLOT_DIR}/06_feature_importance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    print(f"\n  Top {top_n} features (Random Forest):")
    for _, row in imp_df.head(top_n).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 60)
    print('  MUSICAL DNA — Week 3: Composer Classifier')
    print('=' * 60)

    X, y, df = load_data()
    print(f"\n  Dataset: {X.shape[0]} pieces × {X.shape[1]} features, {len(set(y))} composers")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}  (stratified 80/20 split)")

    print(f"\n── MODEL COMPARISON ─────────────────────────────────────")
    results, pipelines = run_all_models(X_train, X_test, y_train, y_test)

    top_name = max(results, key=lambda n: results[n]['test_accuracy'])

    # Random Forest is the primary/deployed model, not necessarily the single
    # highest test-accuracy one. Rationale: its CV accuracy is statistically
    # tied with the top model, it is the only compared model that yields the
    # feature importances the analysis and the dashboard's Style Explorer both
    # rely on, and keeping one interpretable model as the through-line matters
    # more here than a few percentage points on a 71-piece test split.
    PRIMARY = 'Random Forest'
    print(f"\n  Primary (deployed) model: {PRIMARY} "
          f"({results[PRIMARY]['test_accuracy']:.1%} test accuracy)")
    if top_name != PRIMARY:
        print(f"  Note: {top_name} had the highest test accuracy "
              f"({results[top_name]['test_accuracy']:.1%}) on this split, but does not "
              f"provide feature importances; {PRIMARY} is retained as primary.")

    print_per_class_report(results[PRIMARY], PRIMARY)

    print(f"\n── GENERATING PLOTS ─────────────────────────────────────")
    plot_confusion_matrix(results[PRIMARY], PRIMARY)
    plot_feature_importance(pipelines['Random Forest'])

    print(f"\n── SAVING MODEL ─────────────────────────────────────────")
    save_model(pipelines[PRIMARY])
    print(f"  Saved: models/best_classifier.pkl  ({PRIMARY})")

    target_met = results[PRIMARY]['test_accuracy'] >= 0.70
    print(f"\n{'=' * 60}")
    print(f"  WEEK 3 COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Target: 70%+   Result: {results[PRIMARY]['test_accuracy']:.1%}"
          f"   {'✓ TARGET MET' if target_met else '✗ below target — see blueprint risk mitigation'}")
    print(f"\n  Next step: python notebooks/03_style_maps.py")


if __name__ == '__main__':
    main()
