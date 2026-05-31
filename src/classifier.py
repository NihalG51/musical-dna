"""
Musical DNA — Composer Classifier
==================================
Reusable module for training and evaluating composer classification models.
Supports Random Forest, SVM (RBF), and Gradient Boosting via sklearn Pipelines
(StandardScaler + classifier) so scaling is always applied consistently.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .features import FEATURE_NAMES


COMPOSERS = ['bach', 'beethoven', 'chopin', 'debussy', 'mozart', 'rachmaninoff']


def load_data(csv_path='data/processed/features.csv'):
    """Load feature matrix and labels from the extracted CSV.

    Returns:
        X: ndarray (n_pieces, n_features)
        y: ndarray of composer labels
        df: full DataFrame (includes metadata columns)
    """
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df['composer'].to_numpy(dtype=str)
    return X, y, df


def get_models():
    """Return the three classifiers to compare."""
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf', C=10, gamma='scale', random_state=42, probability=True
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }


def build_pipeline(model):
    """Wrap a classifier in a StandardScaler pipeline."""
    return Pipeline([('scaler', StandardScaler()), ('clf', model)])


def evaluate_model(pipeline, X_train, X_test, y_train, y_test, cv=5):
    """Cross-validate on train set, then evaluate on held-out test set.

    Args:
        pipeline: unfitted sklearn Pipeline
        cv: number of StratifiedKFold folds

    Returns:
        dict with cv_mean, cv_std, test_accuracy, classification_report,
        confusion_matrix, y_pred, y_test
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels = sorted(set(y_test))
    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=labels),
        'cm_labels': labels,
        'y_pred': y_pred,
        'y_test': y_test,
    }


def get_feature_importances(pipeline):
    """Extract feature importances from a fitted RF or GB pipeline.

    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted descending,
        or None if the classifier doesn't support feature_importances_.
    """
    clf = pipeline.named_steps['clf']
    if not hasattr(clf, 'feature_importances_'):
        return None
    df = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': clf.feature_importances_})
    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def save_model(pipeline, path='models/best_classifier.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_model(path='models/best_classifier.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
