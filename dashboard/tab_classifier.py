"""Tab 2 — Composer Classifier: upload a MIDI file, see which of the six composers it resembles."""

import os
import pandas as pd
import streamlit as st

from src.features import FEATURE_NAMES, extract_features_from_file
from dashboard.utils import load_composer_classifier, save_uploaded_midi


def render():
    st.header("Composer Classifier")
    st.write(
        "Upload a solo piano/keyboard MIDI file and the Random Forest classifier "
        "(93.0% test accuracy on six composers) will estimate which composer's "
        "style it most resembles. Works best on solo piano MIDI — orchestral or "
        "full-band files will produce noisy, less reliable features."
    )

    uploaded = st.file_uploader("Upload a MIDI file", type=['mid', 'midi'])
    if uploaded is None:
        st.info("Waiting for a .mid or .midi file.")
        return

    tmp_path = save_uploaded_midi(uploaded)
    try:
        with st.spinner("Extracting features and classifying..."):
            features = extract_features_from_file(tmp_path)
            if features is None:
                st.error("Could not parse this file as a MIDI score. Try a different file.")
                return

            pipeline = load_composer_classifier()
            X = [[features[f] for f in FEATURE_NAMES]]
            proba = pipeline.predict_proba(X)[0]
            classes = pipeline.classes_
            prediction = classes[proba.argmax()]

        st.success(f"Predicted composer: **{prediction.title()}** "
                   f"({proba.max():.1%} confidence)")

        proba_df = pd.DataFrame({'Composer': [c.title() for c in classes], 'Probability': proba})
        proba_df = proba_df.sort_values('Probability', ascending=True)
        st.bar_chart(proba_df.set_index('Composer'), horizontal=True)

        with st.expander("Extracted feature values"):
            st.dataframe(
                pd.Series(features, name='Value').to_frame(),
                width='stretch',
            )
    finally:
        os.unlink(tmp_path)
