"""Tab 4 — Similarity Analyzer: upload two MIDI files, get pairwise similarity scores."""

import os
import numpy as np
import pandas as pd
import streamlit as st

from src.similarity import compute_pairwise_similarity
from src.case_model import FEATURE_SETS
from dashboard.utils import load_case_predictor, save_uploaded_midi

METRIC_LABELS = {
    'melodic_similarity': 'Melodic',
    'harmonic_cosine_sim': 'Harmonic',
    'rhythmic_dtw_sim': 'Rhythmic',
    'ngram_overlap_4': 'N-gram',
}


def render():
    st.header("Similarity Analyzer")
    st.write(
        "Upload two MIDI files — a plaintiff's work and a defendant's work — to "
        "compute the same melodic, harmonic, rhythmic, and n-gram overlap scores "
        "used throughout the copyright case dataset."
    )

    col1, col2 = st.columns(2)
    with col1:
        plaintiff_file = st.file_uploader("Plaintiff work (MIDI)", type=['mid', 'midi'], key='plaintiff')
    with col2:
        defendant_file = st.file_uploader("Defendant work (MIDI)", type=['mid', 'midi'], key='defendant')

    if plaintiff_file is None or defendant_file is None:
        st.info("Upload both files to compute similarity.")
        return

    p_path = save_uploaded_midi(plaintiff_file)
    d_path = save_uploaded_midi(defendant_file)
    try:
        with st.spinner("Computing similarity..."):
            scores = compute_pairwise_similarity(p_path, d_path)

        if scores is None:
            st.error("Could not parse one or both files as MIDI scores.")
            return

        cols = st.columns(4)
        for col, (key, label) in zip(cols, METRIC_LABELS.items()):
            col.metric(label, f"{scores[key]:.3f}")

        chart_df = pd.DataFrame({
            'Metric': list(METRIC_LABELS.values()),
            'Score': [scores[k] for k in METRIC_LABELS],
        }).set_index('Metric')
        st.bar_chart(chart_df)

        st.caption(
            "Scores are on a 0-1 scale (1 = identical). See the Copyright Case Explorer "
            "tab for how these scores compare across 33 real cases."
        )

        with st.expander("Estimate infringement-finding probability (illustrative only)"):
            st.warning(
                "This uses the project's logistic regression model, trained on only 33 "
                "cases. Its own analysis found that non-musical case factors predict "
                "outcomes far better than musical similarity (AUC 0.86 vs. 0.39) — so "
                "treat this estimate as illustrative of the model's behavior, not as "
                "legal guidance."
            )
            plaintiff_fame = st.slider("Plaintiff fame (1 = unknown, 5 = global superstar)", 1, 5, 3)
            defendant_success = st.slider("Defendant work commercial success (1-5)", 1, 5, 3)
            expert_testimony = st.checkbox("Expert musicological testimony present")
            us_court = st.checkbox("Formal U.S. federal litigation", value=True)

            if st.button("Estimate"):
                pipeline = load_case_predictor()
                feature_order = FEATURE_SETS['full']
                values = {
                    'melodic_overlap_score': scores['melodic_similarity'],
                    'harmonic_overlap_score': scores['harmonic_cosine_sim'],
                    'rhythmic_overlap_score': scores['rhythmic_dtw_sim'],
                    'ngram_overlap': scores['ngram_overlap_4'],
                    'plaintiff_fame': plaintiff_fame,
                    'defendant_commercial_success': defendant_success,
                    'expert_testimony': int(expert_testimony),
                    'us_court': int(us_court),
                }
                X = np.array([[values[f] for f in feature_order]])
                prob = pipeline.predict_proba(X)[0, 1]
                st.metric("P(infringement found)", f"{prob:.0%}")
    finally:
        os.unlink(p_path)
        os.unlink(d_path)
