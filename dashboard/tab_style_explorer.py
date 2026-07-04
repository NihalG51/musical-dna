"""Tab 1 — Composer Style Explorer: radar chart + feature breakdown for a selected composer."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.classifier import get_feature_importances
from dashboard.utils import load_composer_features, load_composer_classifier

COMPOSER_COLORS = {
    'bach':          '#E74C3C',
    'mozart':        '#3498DB',
    'beethoven':     '#2ECC71',
    'chopin':        '#9B59B6',
    'debussy':       '#F39C12',
    'rachmaninoff':  '#1ABC9C',
}


def _normalize(df, feature_cols):
    norm = df[feature_cols].copy()
    for col in feature_cols:
        fmin, fmax = df[col].min(), df[col].max()
        norm[col] = (df[col] - fmin) / (fmax - fmin) if fmax > fmin else 0.5
    return norm


def _draw_radar(values, feature_labels, color, title):
    n = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = values + values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, vals, alpha=0.20, color=color)
    ax.plot(angles, vals, 'o-', linewidth=2, color=color, markersize=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('_', '\n') for f in feature_labels], size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], size=7, color='grey')
    ax.set_title(title, size=15, fontweight='bold', pad=20, color=color)
    ax.grid(True, alpha=0.3)
    ax.spines['polar'].set_visible(False)
    return fig


def render():
    st.header("Composer Style Explorer")
    st.write(
        "Each classical composer has a distinct computational fingerprint across "
        "21 musical features. Pick a composer to see their profile against the "
        "top 8 features the Random Forest classifier relies on most."
    )

    X, y, df = load_composer_features()
    pipeline = load_composer_classifier()
    top_features = get_feature_importances(pipeline)['feature'].head(8).tolist()

    composers = sorted(df['composer'].unique())
    composer = st.selectbox("Composer", composers, format_func=str.title)

    df_norm = _normalize(df, top_features)
    df_norm['composer'] = df['composer']
    radar_values = df_norm[df_norm['composer'] == composer][top_features].mean().tolist()

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = _draw_radar(radar_values, top_features, COMPOSER_COLORS.get(composer, '#333'),
                           composer.title())
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Feature breakdown")
        st.caption("Raw mean values for this composer vs. the full 353-piece dataset")
        comp_means = df[df['composer'] == composer][top_features].mean()
        overall_means = df[top_features].mean()
        breakdown = comp_means.to_frame(name=composer.title())
        breakdown['Dataset average'] = overall_means
        breakdown['Difference'] = breakdown[composer.title()] - breakdown['Dataset average']
        st.dataframe(
            breakdown.style.format('{:.3f}').background_gradient(
                subset=['Difference'], cmap='RdYlGn', vmin=-breakdown['Difference'].abs().max(),
                vmax=breakdown['Difference'].abs().max(),
            ),
            width='stretch',
        )
        n_pieces = (df['composer'] == composer).sum()
        st.caption(f"Based on {n_pieces} pieces by {composer.title()} in the training dataset.")
