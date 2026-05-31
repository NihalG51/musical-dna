"""
Musical DNA — Copyright Case Dataset Utilities
================================================
Loads, validates, and auto-scores the copyright case CSV.

Auto-scoring: if both midi_plaintiff_path and midi_defendant_path are
filled for a row, compute_pairwise_similarity() fills the three overlap
score columns automatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from .similarity import compute_pairwise_similarity

CASES_CSV = 'data/copyright_cases/cases.csv'

REQUIRED_FIELDS = [
    'case_name', 'year', 'outcome', 'plaintiff_work', 'defendant_work',
    'genre', 'elements_at_issue', 'plaintiff_fame',
    'defendant_commercial_success', 'expert_testimony',
    'jurisdiction', 'settled',
]

SCORE_FIELDS = [
    'melodic_overlap_score',
    'harmonic_overlap_score',
    'rhythmic_overlap_score',
]

VALID_OUTCOMES    = {0, 1}
VALID_BINARY      = {0, 1}
VALID_FAME        = {1, 2, 3, 4, 5}
VALID_ELEMENTS    = {'melody', 'harmony', 'rhythm', 'lyrics', 'groove', 'structure'}


def load_cases(csv_path=CASES_CSV):
    """Load the case CSV into a DataFrame.

    Returns:
        DataFrame with all columns. Score columns are float (NaN if blank).
    """
    df = pd.read_csv(csv_path, dtype=str)

    # Coerce numeric columns
    for col in ['year', 'outcome', 'plaintiff_fame',
                'defendant_commercial_success', 'expert_testimony', 'settled']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in SCORE_FIELDS + ['ngram_overlap']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def validate(df):
    """Check dataset for missing fields, invalid values, and consistency.

    Prints a report and returns a list of issue strings (empty = clean).
    """
    issues = []

    for col in REQUIRED_FIELDS:
        missing = df[col].isna().sum()
        if missing:
            issues.append(f"  MISSING  {col}: {missing} row(s)")

    for i, row in df.iterrows():
        name = row.get('case_name', f'row {i}')

        if not pd.isna(row.get('outcome')) and row['outcome'] not in VALID_OUTCOMES:
            issues.append(f"  INVALID  outcome in '{name}' — must be 0 or 1")

        for f in ['expert_testimony', 'settled']:
            v = row.get(f)
            if not pd.isna(v) and v not in VALID_BINARY:
                issues.append(f"  INVALID  {f} in '{name}' — must be 0 or 1")

        for f in ['plaintiff_fame', 'defendant_commercial_success']:
            v = row.get(f)
            if not pd.isna(v) and v not in VALID_FAME:
                issues.append(f"  INVALID  {f} in '{name}' — must be 1–5")

        if not pd.isna(row.get('elements_at_issue')):
            elements = [e.strip() for e in str(row['elements_at_issue']).split(',')]
            bad = [e for e in elements if e not in VALID_ELEMENTS]
            if bad:
                issues.append(f"  INVALID  elements_at_issue in '{name}': {bad}")

        for col in SCORE_FIELDS:
            v = row.get(col)
            if not pd.isna(v) and not (0.0 <= float(v) <= 1.0):
                issues.append(f"  INVALID  {col} in '{name}' — must be 0.0–1.0")

    score_missing = df[SCORE_FIELDS].isna().all(axis=1).sum()
    if score_missing:
        issues.append(
            f"  INFO     {score_missing} case(s) have no similarity scores yet "
            f"(add MIDI paths and run auto_score, or fill manually)"
        )

    return issues


def auto_score(df, inplace=False):
    """Compute pairwise similarity scores for rows with both MIDI paths set.

    Fills melodic_overlap_score, harmonic_overlap_score, rhythmic_overlap_score,
    and adds ngram_overlap column.

    Args:
        df: case DataFrame from load_cases()
        inplace: if True, modify df in place; otherwise return a copy

    Returns:
        DataFrame with scores filled where MIDI paths exist.
    """
    if not inplace:
        df = df.copy()

    if 'ngram_overlap' not in df.columns:
        df['ngram_overlap'] = np.nan

    scored = 0
    for i, row in df.iterrows():
        p_path = row.get('midi_plaintiff_path', '')
        d_path = row.get('midi_defendant_path', '')

        if not isinstance(p_path, str) or not p_path.strip():
            continue
        if not isinstance(d_path, str) or not d_path.strip():
            continue

        if not Path(p_path).exists():
            print(f"  Warning: plaintiff MIDI not found: {p_path}")
            continue
        if not Path(d_path).exists():
            print(f"  Warning: defendant MIDI not found: {d_path}")
            continue

        print(f"  Scoring: {row['case_name']}...", end=' ', flush=True)
        scores = compute_pairwise_similarity(p_path, d_path)

        if scores is None:
            print("FAILED")
            continue

        df.at[i, 'melodic_overlap_score']  = scores['melodic_similarity']
        df.at[i, 'harmonic_overlap_score'] = scores['harmonic_cosine_sim']
        df.at[i, 'rhythmic_overlap_score'] = scores['rhythmic_dtw_sim']
        df.at[i, 'ngram_overlap']          = scores['ngram_overlap_4']
        scored += 1
        print(f"melodic={scores['melodic_similarity']:.3f}  "
              f"harmonic={scores['harmonic_cosine_sim']:.3f}  "
              f"rhythmic={scores['rhythmic_dtw_sim']:.3f}  "
              f"ngram={scores['ngram_overlap_4']:.3f}")

    print(f"\n  Auto-scored {scored} case(s).")
    return df


def save_cases(df, csv_path=CASES_CSV):
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def print_summary(df):
    total = len(df)
    scored = df[SCORE_FIELDS].notna().all(axis=1).sum()
    infringement = int(df['outcome'].sum())

    print(f"\n  Cases total:       {total}")
    print(f"  Infringement (1):  {infringement}  ({infringement/total:.0%})")
    print(f"  No infringement:   {total - infringement}  ({(total-infringement)/total:.0%})")
    print(f"  Fully scored:      {scored} / {total}")

    if 'genre' in df.columns:
        print(f"\n  By genre:")
        for genre, count in df['genre'].value_counts().items():
            print(f"    {genre:<20} {count:3d}")

    if 'jurisdiction' in df.columns:
        print(f"\n  By jurisdiction:")
        for jx, count in df['jurisdiction'].value_counts().items():
            print(f"    {jx:<25} {count:3d}")
