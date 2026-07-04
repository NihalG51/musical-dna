"""
Musical DNA — Week 8: Streamlit Dashboard
============================================
Four tabs:
  1. Composer Style Explorer — radar chart + feature breakdown per composer
  2. Composer Classifier — upload a MIDI file, see which composer it matches
  3. Copyright Case Explorer — interactive, filterable table of 43 cases
  4. Similarity Analyzer — upload two MIDI files, get a similarity score

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard import tab_style_explorer, tab_classifier, tab_case_explorer, tab_similarity

st.set_page_config(page_title="Musical DNA", page_icon="🎼", layout="wide")

st.title("🎼 Musical DNA")
st.caption(
    "Computational fingerprinting of compositional style, applied to classical "
    "composer classification and music copyright case analysis."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Composer Style Explorer",
    "Composer Classifier",
    "Copyright Case Explorer",
    "Similarity Analyzer",
])

with tab1:
    tab_style_explorer.render()

with tab2:
    tab_classifier.render()

with tab3:
    tab_case_explorer.render()

with tab4:
    tab_similarity.render()

st.divider()
st.caption(
    "Musical DNA is an independent research project pairing a 21-feature style "
    "fingerprinting engine (trained on 353 pieces by six classical composers) with "
    "an original dataset of 43 real music copyright cases. See paper/musical_dna_draft.md "
    "in the project repository for full methodology, results, and limitations."
)
