"""
Musical DNA — Copyright Case Outcome Predictor
================================================
Reusable module for training and evaluating logistic regression models that
predict case outcome (infringement found vs not) from musical similarity
scores and non-musical case factors.

With only ~33 scored cases, a single train/test split is too noisy to trust —
evaluation here relies on cross-validation (stratified k-fold and
leave-one-out) rather than a held-out test set.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate

from .cases import load_cases, SCORE_FIELDS

MUSICAL_FEATURES = [
    'melodic_overlap_score', 'harmonic_overlap_score',
    'rhythmic_overlap_score', 'ngram_overlap',
]

NON_MUSICAL_FEATURES = [
    'plaintiff_fame', 'defendant_commercial_success',
    'expert_testimony', 'us_court',
]
# 'settled' is deliberately excluded as a predictor: for every settled case in
# this dataset, outcome was coded as 1 ("infringement implied") by definition
# at data-entry time, since there was no independent court ruling to measure.
# settled=1 -> outcome=1 with perfect consistency (13/13) — using it as a
# feature would be predicting the label from itself, not a genuine legal signal.

FEATURE_SETS = {
    'musical':     MUSICAL_FEATURES,
    'non_musical': NON_MUSICAL_FEATURES,
    'full':        MUSICAL_FEATURES + NON_MUSICAL_FEATURES,
}

# Jurisdiction values naming a formal US federal court (circuit, district, or
# the Supreme Court). Everything else (UK/Australia courts, settled pre-suit
# or out-of-court, "Unknown/Settled...") is treated as 0. Jurisdiction itself
# has 18 near-unique values across 33 rows — far too sparse to one-hot encode
# — so this collapses it to the one distinction the research question cares
# about: did this go through formal US litigation, or not.
_US_COURT_MARKERS = ('Circuit', 'C.D.', 'N.D.', 'S.D.', 'D.', 'Supreme Court')


def derive_us_court(jurisdiction):
    if not isinstance(jurisdiction, str):
        return 0
    return int(any(marker in jurisdiction for marker in _US_COURT_MARKERS))


def load_case_features(feature_set='full', csv_path=None):
    """Load the scored subset of the case dataset as (X, y, df).

    Args:
        feature_set: one of 'musical', 'non_musical', 'full'
        csv_path: optional override for the cases CSV path

    Returns:
        X: ndarray (n_cases, n_features)
        y: ndarray of outcomes (0/1)
        df: filtered DataFrame (only fully-scored rows), with 'us_court' added
    """
    kwargs = {'csv_path': csv_path} if csv_path else {}
    df = load_cases(**kwargs)
    df = df[df[SCORE_FIELDS].notna().all(axis=1)].copy()
    df['us_court'] = df['jurisdiction'].apply(derive_us_court)

    features = FEATURE_SETS[feature_set]
    X = df[features].to_numpy(dtype=float)
    y = df['outcome'].to_numpy(dtype=int)
    return X, y, df


def build_pipeline(C=1.0):
    """Standardize features, then L2-regularized logistic regression.

    Regularization matters more than usual here: with as few as 4-9 features
    over 33 rows, an unregularized fit can separate the classes trivially.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=C, max_iter=2000, random_state=42)),
    ])


def evaluate_model(pipeline, X, y, n_splits=5):
    """Cross-validate a pipeline two ways and return both estimates.

    Stratified k-fold is the standard estimate, but with only 33 rows each
    fold holds ~6-7 cases — leave-one-out is included alongside it as a
    lower-variance (if slightly optimistic) estimate for a dataset this size.
    """
    scoring = ['accuracy', 'precision', 'recall', 'roc_auc']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    kfold_scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)

    loo = LeaveOneOut()
    loo_scores = cross_validate(pipeline, X, y, cv=loo, scoring=['accuracy'])

    return {
        'kfold': {m: (kfold_scores[f'test_{m}'].mean(), kfold_scores[f'test_{m}'].std())
                  for m in scoring},
        'loo_accuracy': loo_scores['test_accuracy'].mean(),
    }


def get_coefficients(pipeline, feature_names):
    """Fit-in-place pipeline's standardized coefficients as odds ratios.

    Coefficients are on standardized features, so magnitude is directly
    comparable across features regardless of their original scale.
    """
    clf = pipeline.named_steps['clf']
    coefs = clf.coef_[0]
    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'odds_ratio': np.exp(coefs),
    })
    return df.reindex(df['coefficient'].abs().sort_values(ascending=False).index)


def loo_predictions(pipeline, X, y):
    """Out-of-fold predicted probabilities for every row via leave-one-out.

    Used to flag outlier cases (model confidently predicts one outcome, court
    ruled the other) without the leakage of scoring a row the model was
    trained on.
    """
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        pipeline.fit(X[train_idx], y[train_idx])
        probs[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
    return probs


def save_model(pipeline, path='models/case_predictor.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
