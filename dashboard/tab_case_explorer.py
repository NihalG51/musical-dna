"""Tab 3 — Copyright Case Explorer: interactive, filterable table of the case dataset."""

import streamlit as st

from dashboard.utils import load_case_dataset

DISPLAY_COLS = [
    'case_name', 'year', 'outcome', 'plaintiff_work', 'defendant_work', 'genre',
    'elements_at_issue', 'jurisdiction', 'settled', 'is_scored',
    'melodic_overlap_score', 'harmonic_overlap_score', 'rhythmic_overlap_score',
]


def render():
    st.header("Copyright Case Explorer")
    st.write(
        "43 hand-curated music copyright cases spanning 1946–2023, 33 of which are "
        "scored with computed melodic/harmonic/rhythmic similarity. Filter below to "
        "explore the dataset."
    )

    df = load_case_dataset()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        genres = st.multiselect("Genre", sorted(df['genre'].dropna().unique()))
    with col2:
        outcome_filter = st.selectbox("Outcome", ["All", "Infringement found", "No infringement"])
    with col3:
        scored_only = st.checkbox("Scored cases only", value=False)
    with col4:
        year_min, year_max = int(df['year'].min()), int(df['year'].max())
        year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))

    filtered = df.copy()
    if genres:
        filtered = filtered[filtered['genre'].isin(genres)]
    if outcome_filter == "Infringement found":
        filtered = filtered[filtered['outcome'] == 1]
    elif outcome_filter == "No infringement":
        filtered = filtered[filtered['outcome'] == 0]
    if scored_only:
        filtered = filtered[filtered['is_scored']]
    filtered = filtered[(filtered['year'] >= year_range[0]) & (filtered['year'] <= year_range[1])]

    m1, m2, m3 = st.columns(3)
    m1.metric("Cases shown", len(filtered))
    infringe_rate = filtered['outcome'].mean() if len(filtered) else 0
    m2.metric("Infringement rate", f"{infringe_rate:.0%}")
    m3.metric("Scored", int(filtered['is_scored'].sum()))

    st.dataframe(
        filtered[DISPLAY_COLS].sort_values('year'),
        width='stretch',
        hide_index=True,
        column_config={
            'outcome': st.column_config.NumberColumn('Infringement?', format='%d'),
            'melodic_overlap_score': st.column_config.ProgressColumn(
                'Melodic', min_value=0, max_value=1, format='%.2f'),
            'harmonic_overlap_score': st.column_config.ProgressColumn(
                'Harmonic', min_value=0, max_value=1, format='%.2f'),
            'rhythmic_overlap_score': st.column_config.ProgressColumn(
                'Rhythmic', min_value=0, max_value=1, format='%.2f'),
        },
    )

    with st.expander("Selected case notes"):
        case_names = filtered['case_name'].tolist()
        if case_names:
            selected = st.selectbox("Case", case_names)
            row = filtered[filtered['case_name'] == selected].iloc[0]
            st.write(row['notes'])
        else:
            st.write("No cases match the current filters.")
