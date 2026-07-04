"""
Musical DNA — Dashboard shared utilities
==========================================
Cached loaders and small helpers shared across the four dashboard tabs.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import load_data as _load_composer_data, COMPOSERS
from src.features import FEATURE_NAMES, extract_features_from_file
from src.cases import load_cases, SCORE_FIELDS
from src.similarity import compute_pairwise_similarity
from src.case_model import FEATURE_SETS, derive_us_court


@st.cache_data
def load_composer_features():
    """Cached load of the 353-piece composer feature matrix."""
    X, y, df = _load_composer_data()
    return X, y, df


@st.cache_resource
def load_composer_classifier():
    with open('models/best_classifier.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_case_predictor():
    with open('models/case_predictor.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_case_dataset():
    """Cached load of the copyright case CSV, with a derived scored flag."""
    df = load_cases()
    df['is_scored'] = df[SCORE_FIELDS].notna().all(axis=1)
    df['us_court'] = df['jurisdiction'].apply(derive_us_court)
    return df


def save_uploaded_midi(uploaded_file):
    """Persist a Streamlit UploadedFile to a temp path music21 can read from.

    music21's converter.parse() needs a real file path, not a file-like
    object, so every uploaded MIDI has to be written to disk first.
    """
    suffix = Path(uploaded_file.name).suffix or '.mid'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def normalize_to_dataset(value, series):
    """Min-max normalize a single value against a reference column's range."""
    vmin, vmax = series.min(), series.max()
    if vmax <= vmin:
        return 0.5
    return float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
