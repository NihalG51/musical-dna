"""
Musical DNA — Week 7: Outlier Cases & Component A/B Synthesis
===============================================================
Two analyses:

1. Outlier identification: which cases most defy the "similarity predicts
   outcome" intuition — high computed musical similarity with no
   infringement found, or low similarity with infringement found (real
   verdicts only, since settled cases have no independent ruling to
   compare against). Cases with an exact-zero rhythmic score are excluded
   from candidacy — that floor is a known trim-to-90s pipeline artifact
   (see notes on Williams v. Broadus / Larrikin v. EMI in cases.csv), not a
   genuine absence of rhythmic similarity, so they'd be a misleading
   "low similarity" exemplar.

2. Synthesis: Component A's 21-feature style-fingerprinting engine (built
   for classifying classical composers) applied directly to the plaintiff/
   defendant works in the 5 selected outlier cases. Each piece's 21-feature
   vector is standardized against the classical-composer feature
   distribution, then compared via cosine similarity — a completely
   independent measure of "do these two pieces occupy similar territory in
   general compositional-style space," distinct from the melodic/harmonic/
   rhythmic/n-gram similarity engine built specifically for Component B.

Usage:
    python notebooks/07_outlier_analysis.py

Reads:  data/copyright_cases/cases.csv
        data/processed/features.csv (reference distribution for standardizing)
        data/midi/copyright_cases/<case>/{plaintiff,defendant}.mid
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.case_model import load_case_features
from src.features import FEATURE_NAMES, extract_features_from_file

N_OUTLIERS_PER_DIRECTION = {'no_infringement': 3, 'infringement': 2}


# =============================================================================
# OUTLIER IDENTIFICATION
# =============================================================================

def select_outliers(df):
    df = df.copy()
    df['musical_composite'] = df[
        ['melodic_overlap_score', 'harmonic_overlap_score', 'rhythmic_overlap_score']
    ].mean(axis=1)

    # Exclude exact-zero rhythmic scores — a documented pipeline artifact,
    # not a genuine "no rhythmic similarity" finding.
    clean = df[df['rhythmic_overlap_score'] > 0.0]

    high_sim_no_infringe = (
        clean[clean['outcome'] == 0]
        .sort_values('musical_composite', ascending=False)
        .head(N_OUTLIERS_PER_DIRECTION['no_infringement'])
    )
    # Settled cases have no independent ruling to compare a low similarity
    # score against — restrict this direction to cases that went to a real
    # verdict.
    low_sim_infringe = (
        clean[(clean['outcome'] == 1) & (clean['settled'] == 0)]
        .sort_values('musical_composite')
        .head(N_OUTLIERS_PER_DIRECTION['infringement'])
    )

    print(f"\n── OUTLIER CASES ─────────────────────────────────────────")
    print(f"  High musical similarity, but no infringement found:")
    print(f"  {'Case':<42} {'Composite':>10} {'Mel':>6} {'Harm':>6} {'Rhy':>6}")
    print('  ' + '-' * 74)
    for _, row in high_sim_no_infringe.iterrows():
        print(f"  {row['case_name']:<42} {row['musical_composite']:>10.3f} "
              f"{row['melodic_overlap_score']:>6.2f} {row['harmonic_overlap_score']:>6.2f} "
              f"{row['rhythmic_overlap_score']:>6.2f}")

    print(f"\n  Low musical similarity, but infringement found (real verdicts only):")
    print(f"  {'Case':<42} {'Composite':>10} {'Mel':>6} {'Harm':>6} {'Rhy':>6}")
    print('  ' + '-' * 74)
    for _, row in low_sim_infringe.iterrows():
        print(f"  {row['case_name']:<42} {row['musical_composite']:>10.3f} "
              f"{row['melodic_overlap_score']:>6.2f} {row['harmonic_overlap_score']:>6.2f} "
              f"{row['rhythmic_overlap_score']:>6.2f}")

    return pd.concat([high_sim_no_infringe, low_sim_infringe])


# =============================================================================
# COMPONENT A / B SYNTHESIS
# =============================================================================

def fit_reference_scaler(features_csv='data/processed/features.csv'):
    """Fit a StandardScaler on the classical-composer feature distribution
    (Component A's training data) — this defines the "style space" that
    the copyright-case works get projected into for comparison.
    """
    ref = pd.read_csv(features_csv)
    scaler = StandardScaler()
    scaler.fit(ref[FEATURE_NAMES].to_numpy(dtype=float))
    return scaler


def style_fingerprint_similarity(plaintiff_mid, defendant_mid, scaler):
    """Extract Component A's 21 features from both MIDI files, standardize
    against the classical-composer distribution, and return cosine
    similarity between the two standardized vectors.
    """
    p_feats = extract_features_from_file(plaintiff_mid)
    d_feats = extract_features_from_file(defendant_mid)
    if p_feats is None or d_feats is None:
        return None

    p_vec = np.array([[p_feats[f] for f in FEATURE_NAMES]])
    d_vec = np.array([[d_feats[f] for f in FEATURE_NAMES]])
    p_std = scaler.transform(p_vec)
    d_std = scaler.transform(d_vec)
    return float(cosine_similarity(p_std, d_std)[0, 0])


def run_synthesis(outlier_df):
    scaler = fit_reference_scaler()

    print(f"\n── SYNTHESIS: Component A style-fingerprint vs Component B similarity ──")
    print(f"  Applying the 21-feature classical-composer engine directly to the")
    print(f"  disputed works — an independent style-space comparison alongside")
    print(f"  Component B's purpose-built melodic/harmonic/rhythmic engine.\n")
    print(f"  {'Case':<42} {'Style-FP cos-sim':>17} {'Mel':>6} {'Harm':>6} {'Rhy':>6} {'Ruling':>10}")
    print('  ' + '-' * 90)

    rows = []
    for _, row in outlier_df.iterrows():
        sim = style_fingerprint_similarity(
            row['midi_plaintiff_path'], row['midi_defendant_path'], scaler
        )
        ruling = 'Infringe' if row['outcome'] == 1 else 'No infr.'
        rows.append({'case_name': row['case_name'], 'style_fp_sim': sim,
                     'melodic': row['melodic_overlap_score'],
                     'harmonic': row['harmonic_overlap_score'],
                     'rhythmic': row['rhythmic_overlap_score'],
                     'outcome': row['outcome']})
        print(f"  {row['case_name']:<42} {sim:>17.3f} "
              f"{row['melodic_overlap_score']:>6.2f} {row['harmonic_overlap_score']:>6.2f} "
              f"{row['rhythmic_overlap_score']:>6.2f} {ruling:>10}")

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 60)
    print('  MUSICAL DNA — Week 7: Outlier Cases & A/B Synthesis')
    print('=' * 60)

    _, _, df = load_case_features(feature_set='full')
    outliers = select_outliers(df)
    synthesis = run_synthesis(outliers)

    print(f"\n{'=' * 60}")
    print(f"  WEEK 7 MILESTONE")
    print(f"{'=' * 60}")
    print(f"\n  {len(outliers)} outlier cases identified and cross-checked against")
    print(f"  Component A's independent style-fingerprint engine.")
    print(f"  Next step: write up case studies + legal implications in the paper.")

    return outliers, synthesis


if __name__ == '__main__':
    main()
