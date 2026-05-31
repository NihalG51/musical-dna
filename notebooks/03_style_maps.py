"""
Musical DNA — Week 4: Style Maps & Visualizations
===================================================
Generates the project's signature visuals:
  1. Composer style fingerprint radar charts (top 8 RF features)
  2. t-SNE scatter — all 353 pieces projected to 2D, colored by composer
  3. PCA scatter — same data, faster and more interpretable axes

Usage:
    python notebooks/03_style_maps.py

Reads:  data/processed/features.csv
        models/best_classifier.pkl
Saves:  notebooks/plots/07_style_fingerprints.png
        notebooks/plots/07b_style_fingerprints_overlay.png
        notebooks/plots/08_tsne.png
        notebooks/plots/09_pca.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import load_data, load_model, get_feature_importances

PLOT_DIR = 'notebooks/plots'
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

COMPOSER_ORDER = ['bach', 'mozart', 'beethoven', 'chopin', 'debussy', 'rachmaninoff']
COMPOSER_COLORS = {
    'bach':          '#E74C3C',
    'mozart':        '#3498DB',
    'beethoven':     '#2ECC71',
    'chopin':        '#9B59B6',
    'debussy':       '#F39C12',
    'rachmaninoff':  '#1ABC9C',
}
COMPOSER_MARKERS = {
    'bach': 'o', 'mozart': 's', 'beethoven': '^',
    'chopin': 'D', 'debussy': 'P', 'rachmaninoff': '*',
}


# =============================================================================
# SHARED HELPERS
# =============================================================================

def top_features_by_importance(pipeline, n=8):
    """Return top-N feature names from the fitted RF pipeline."""
    imp_df = get_feature_importances(pipeline)
    return imp_df['feature'].head(n).tolist()


def normalize_feature_matrix(df, feature_cols):
    """Min-max normalize each feature column to [0, 1] across the full dataset."""
    norm = df[feature_cols].copy()
    for col in feature_cols:
        fmin, fmax = df[col].min(), df[col].max()
        norm[col] = (df[col] - fmin) / (fmax - fmin) if fmax > fmin else 0.5
    return norm


def composer_radar_values(df_norm, feature_cols):
    """Return dict of {composer: [mean normalized values]} for radar plotting."""
    return {
        comp: df_norm[df_norm['composer'] == comp][feature_cols].mean().tolist()
        for comp in COMPOSER_ORDER
        if comp in df_norm['composer'].values
    }


# =============================================================================
# PLOT 1: INDIVIDUAL RADAR CHARTS (2×3 GRID)
# =============================================================================

def _draw_radar(ax, values, feature_labels, color, title):
    n = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = values + values[:1]

    ax.fill(angles, vals, alpha=0.20, color=color)
    ax.plot(angles, vals, 'o-', linewidth=2, color=color, markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f.replace('_', '\n') for f in feature_labels],
        size=7.5, color='#333333',
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], size=6, color='grey')
    ax.set_title(title, size=14, fontweight='bold', pad=18, color=color)
    ax.grid(True, alpha=0.3)
    ax.spines['polar'].set_visible(False)


def plot_style_fingerprints(df, top_features):
    """2×3 grid of individual composer radar charts."""
    df_norm = normalize_feature_matrix(df, top_features)
    df_norm['composer'] = df['composer']
    radar_vals = composer_radar_values(df_norm, top_features)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for i, comp in enumerate(COMPOSER_ORDER):
        _draw_radar(
            axes[i], radar_vals[comp], top_features,
            color=COMPOSER_COLORS[comp], title=comp.title(),
        )

    fig.suptitle(
        'Composer Style Fingerprints\n(top 8 features by Random Forest importance, normalized 0–1)',
        fontsize=15, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    path = f'{PLOT_DIR}/07_style_fingerprints.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 1b: OVERLAY RADAR (ALL COMPOSERS ON ONE CHART)
# =============================================================================

def plot_style_fingerprints_overlay(df, top_features):
    """All six composers overlaid on a single radar for direct comparison."""
    df_norm = normalize_feature_matrix(df, top_features)
    df_norm['composer'] = df['composer']
    radar_vals = composer_radar_values(df_norm, top_features)

    n = len(top_features)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    labels = [f.replace('_', '\n') for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for comp in COMPOSER_ORDER:
        vals = radar_vals[comp] + radar_vals[comp][:1]
        color = COMPOSER_COLORS[comp]
        ax.fill(angles, vals, alpha=0.08, color=color)
        ax.plot(angles, vals, 'o-', linewidth=2, color=color, markersize=5, label=comp.title())

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], size=8, color='grey')
    ax.grid(True, alpha=0.3)
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
    ax.set_title(
        'All Composer Style Fingerprints Overlaid',
        size=15, fontweight='bold', pad=25,
    )

    path = f'{PLOT_DIR}/07b_style_fingerprints_overlay.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 2: t-SNE
# =============================================================================

def plot_tsne(X, y):
    """2D t-SNE projection of all 353 pieces, colored and shaped by composer."""
    print("  Running t-SNE (this takes ~30 seconds)...", end=' ', flush=True)

    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X_scaled)
    print("done.")

    fig, ax = plt.subplots(figsize=(10, 8))

    for comp in COMPOSER_ORDER:
        mask = y == comp
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=COMPOSER_COLORS[comp],
            marker=COMPOSER_MARKERS[comp],
            s=60, alpha=0.75, edgecolors='white', linewidths=0.4,
            label=comp.title(), zorder=3,
        )

    # Composer centroids
    for comp in COMPOSER_ORDER:
        mask = y == comp
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, c=COMPOSER_COLORS[comp], s=250,
                   marker='X', edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(comp.title(), (cx, cy), fontsize=10, fontweight='bold',
                    ha='center', va='bottom', xytext=(0, 10),
                    textcoords='offset points', color=COMPOSER_COLORS[comp])

    ax.set_title('t-SNE Projection of 353 Pieces\n(perplexity=30, X markers = composer centroids)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = f'{PLOT_DIR}/08_tsne.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 3: PCA
# =============================================================================

def plot_pca(X, y):
    """2D PCA projection — faster than t-SNE, axes are interpretable."""
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var1, var2 = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))

    for comp in COMPOSER_ORDER:
        mask = y == comp
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=COMPOSER_COLORS[comp],
            marker=COMPOSER_MARKERS[comp],
            s=60, alpha=0.75, edgecolors='white', linewidths=0.4,
            label=comp.title(), zorder=3,
        )

    for comp in COMPOSER_ORDER:
        mask = y == comp
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, c=COMPOSER_COLORS[comp], s=250,
                   marker='X', edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(comp.title(), (cx, cy), fontsize=10, fontweight='bold',
                    ha='center', va='bottom', xytext=(0, 10),
                    textcoords='offset points', color=COMPOSER_COLORS[comp])

    ax.set_title(
        f'PCA Projection of 353 Pieces\n'
        f'PC1 {var1:.1%} variance  |  PC2 {var2:.1%} variance  |  X = centroid',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlabel(f'PC1 ({var1:.1%} variance)')
    ax.set_ylabel(f'PC2 ({var2:.1%} variance)')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = f'{PLOT_DIR}/09_pca.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 60)
    print('  MUSICAL DNA — Week 4: Style Maps')
    print('=' * 60)

    X, y, df = load_data()
    print(f"\n  Dataset: {X.shape[0]} pieces, {X.shape[1]} features")

    print("\n  Loading best classifier to get feature importances...")
    pipeline = load_model()
    top_features = top_features_by_importance(pipeline, n=8)
    print(f"  Top 8 features (RF importances):")
    for f in top_features:
        print(f"    {f}")

    print("\n── RADAR CHARTS ─────────────────────────────────────────")
    plot_style_fingerprints(df, top_features)
    plot_style_fingerprints_overlay(df, top_features)

    print("\n── DIMENSIONALITY REDUCTION ─────────────────────────────")
    plot_tsne(X, y)
    plot_pca(X, y)

    print(f"\n{'=' * 60}")
    print(f"  WEEK 4 (style maps) COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Plots saved to {PLOT_DIR}/")
    print(f"\n  Remaining Week 4 tasks (manual):")
    print(f"    - Download 5-10 AI-generated MIDI files from Suno/Udio")
    print(f"      ('in the style of Bach', 'in the style of Chopin', etc.)")
    print(f"    - Run classifier on them:")
    print(f"        from src.classifier import load_model, load_data")
    print(f"        from src.features import extract_features_from_file")
    print(f"        pipeline = load_model()")
    print(f"        features = extract_features_from_file('path/to/ai.mid')")
    print(f"        prediction = pipeline.predict([list(features.values())])")
    print(f"    - Write up Component A findings (3-4 pages)")


if __name__ == '__main__':
    main()
