"""
Musical DNA — Data Exploration & Sanity Check
===============================================
Generates visualizations to verify that extracted features
make musical sense before building the classifier.

Usage:
    python notebooks/01_explore.py

Reads:  data/processed/features.csv
Saves:  notebooks/plots/ (all generated figures)

What to look for:
  - Do feature distributions differ between composers?
  - Do the differences match musical intuition?
  - Are there any features with zero variance or obvious bugs?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────
CSV_PATH = 'data/processed/features.csv'
PLOT_DIR = 'notebooks/plots'
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

# Nice color palette for 6 composers
COMPOSER_COLORS = {
    'bach':          '#E74C3C',   # red
    'mozart':        '#3498DB',   # blue
    'beethoven':     '#2ECC71',   # green
    'chopin':        '#9B59B6',   # purple
    'debussy':       '#F39C12',   # orange
    'rachmaninoff':  '#1ABC9C',   # teal
}

COMPOSER_ORDER = ['bach', 'mozart', 'beethoven', 'chopin', 'debussy', 'rachmaninoff']


def load_data():
    """Load the feature CSV and print basic stats."""
    df = pd.read_csv(CSV_PATH)
    
    print("=" * 60)
    print("  MUSICAL DNA — Data Exploration")
    print("=" * 60)
    print(f"\n  Dataset: {len(df)} pieces, {len(df.columns)} columns")
    print(f"\n  Per-composer counts:")
    for comp in COMPOSER_ORDER:
        count = len(df[df['composer'] == comp])
        print(f"    {comp:20s} {count:4d} pieces")
    
    # Identify feature columns (exclude metadata)
    meta_cols = ['composer', 'filename', 'filepath']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"\n  Feature columns ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"    {f}")
    
    return df, feature_cols


def check_data_quality(df, feature_cols):
    """Check for missing values, zero-variance features, and outliers."""
    print("\n\n── DATA QUALITY CHECK ──────────────────────────────────")
    
    issues = []
    
    for col in feature_cols:
        missing = df[col].isna().sum()
        zeros = (df[col] == 0).sum()
        variance = df[col].var()
        
        if missing > 0:
            issues.append(f"  WARNING: {col} has {missing} missing values")
        if variance == 0:
            issues.append(f"  WARNING: {col} has zero variance (all same value)")
        if zeros > len(df) * 0.8:
            issues.append(f"  WARNING: {col} is 0 for {zeros}/{len(df)} pieces ({zeros/len(df)*100:.0f}%)")
    
    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(issue)
    else:
        print("  All features look clean — no missing values or zero-variance columns.")
    
    # Print summary statistics
    print("\n  Feature summary statistics:")
    summary = df[feature_cols].describe().T[['mean', 'std', 'min', 'max']]
    print(summary.to_string())
    
    return issues


def plot_histograms(df, feature_cols):
    """Plot histograms of each feature, color-coded by composer.
    
    THIS IS YOUR MAIN SANITY CHECK.
    Look for features where composers clearly separate.
    """
    print("\n\n── PLOTTING HISTOGRAMS ─────────────────────────────────")
    
    # Select features to plot (skip filepath, filename, etc.)
    plot_features = [f for f in feature_cols if df[f].var() > 0]
    
    n_features = len(plot_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(plot_features):
        ax = axes[i]
        
        for comp in COMPOSER_ORDER:
            data = df[df['composer'] == comp][feature].dropna()
            if len(data) > 0:
                ax.hist(data, bins=20, alpha=0.5, label=comp.title(),
                       color=COMPOSER_COLORS[comp], density=True)
        
        ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Feature Distributions by Composer', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    save_path = f'{PLOT_DIR}/01_feature_histograms.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_boxplots(df, feature_cols):
    """Box plots comparing each feature across composers."""
    print("  Generating box plots...")
    
    plot_features = [f for f in feature_cols if df[f].var() > 0]
    
    n_features = len(plot_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    palette = [COMPOSER_COLORS[c] for c in COMPOSER_ORDER]
    
    for i, feature in enumerate(plot_features):
        ax = axes[i]
        
        plot_df = df[['composer', feature]].dropna()
        plot_df = plot_df[plot_df['composer'].isin(COMPOSER_ORDER)]
        
        sns.boxplot(data=plot_df, x='composer', y=feature, ax=ax,
                   order=COMPOSER_ORDER, palette=palette, width=0.6)
        
        ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30, labelsize=9)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Feature Comparison Across Composers', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    save_path = f'{PLOT_DIR}/02_feature_boxplots.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_matrix(df, feature_cols):
    """Correlation matrix of all features — helps spot redundant features."""
    print("  Generating correlation matrix...")
    
    numeric_features = [f for f in feature_cols if df[f].var() > 0]
    corr = df[numeric_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               square=True, ax=ax, annot_kws={'size': 7},
               xticklabels=[f.replace('_', '\n') for f in numeric_features],
               yticklabels=[f.replace('_', ' ') for f in numeric_features])
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = f'{PLOT_DIR}/03_correlation_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_radar_preview(df, feature_cols):
    """Preview radar charts — one per composer showing normalized feature means.
    
    These will become the project's signature visual.
    """
    print("  Generating radar chart preview...")
    
    # Pick the most interesting features for the radar (up to 8)
    # Prioritize features with high between-composer variance
    numeric_features = [f for f in feature_cols if df[f].var() > 0]
    
    # Calculate between-composer variance for each feature
    between_var = {}
    for f in numeric_features:
        means = df.groupby('composer')[f].mean()
        between_var[f] = means.var()
    
    # Pick top 8 most discriminating features
    top_features = sorted(between_var, key=between_var.get, reverse=True)[:8]
    
    print(f"    Top 8 most discriminating features:")
    for f in top_features:
        print(f"      {f}")
    
    # Normalize each feature to 0-1 range
    radar_data = {}
    for comp in COMPOSER_ORDER:
        comp_data = df[df['composer'] == comp]
        values = []
        for f in top_features:
            val = comp_data[f].mean()
            fmin = df[f].min()
            fmax = df[f].max()
            if fmax > fmin:
                normalized = (val - fmin) / (fmax - fmin)
            else:
                normalized = 0.5
            values.append(normalized)
        radar_data[comp] = values
    
    # Plot
    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    labels = [f.replace('_', '\n') for f in top_features]
    
    for i, comp in enumerate(COMPOSER_ORDER):
        ax = axes[i]
        values = radar_data[comp] + radar_data[comp][:1]  # close the polygon
        
        ax.fill(angles, values, alpha=0.25, color=COMPOSER_COLORS[comp])
        ax.plot(angles, values, 'o-', linewidth=2, color=COMPOSER_COLORS[comp], markersize=5)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(comp.title(), size=14, fontweight='bold', pad=20,
                    color=COMPOSER_COLORS[comp])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Composer Style Fingerprints (Preview)', fontsize=16,
                fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = f'{PLOT_DIR}/04_radar_preview.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def musical_sanity_check(df):
    """Print specific musical predictions and check if the data agrees.
    
    These are things you KNOW should be true from music theory.
    If the data disagrees, something might be wrong with the feature.
    """
    print("\n\n── MUSICAL SANITY CHECK ────────────────────────────────")
    print("  Checking if features match musical intuition...\n")
    
    means = df.groupby('composer').mean(numeric_only=True)
    
    checks = []
    
    # Check 1: Debussy should have lower key stability than Bach
    if 'key_stability' in means.columns:
        bach_ks = means.loc['bach', 'key_stability']
        debussy_ks = means.loc['debussy', 'key_stability']
        result = "PASS" if bach_ks > debussy_ks else "INVESTIGATE"
        checks.append(f"  [{result}] Key stability: Bach ({bach_ks:.3f}) > Debussy ({debussy_ks:.3f})")
        checks.append(f"         Bach's music is firmly tonal; Debussy breaks tonal rules.")
    
    # Check 2: Chopin should have higher note density than Mozart
    if 'note_density' in means.columns:
        chopin_nd = means.loc['chopin', 'note_density']
        mozart_nd = means.loc['mozart', 'note_density']
        result = "PASS" if chopin_nd > mozart_nd else "INVESTIGATE"
        checks.append(f"  [{result}] Note density: Chopin ({chopin_nd:.3f}) > Mozart ({mozart_nd:.3f})")
        checks.append(f"         Chopin writes virtuosic ornamental passages; Mozart is more classical.")
    
    # Check 3: Rachmaninoff should have wider pitch range than Mozart
    if 'pitch_range' in means.columns:
        rach_pr = means.loc['rachmaninoff', 'pitch_range']
        mozart_pr = means.loc['mozart', 'pitch_range']
        result = "PASS" if rach_pr > mozart_pr else "INVESTIGATE"
        checks.append(f"  [{result}] Pitch range: Rachmaninoff ({rach_pr:.1f}) > Mozart ({mozart_pr:.1f})")
        checks.append(f"         Rachmaninoff exploits the full keyboard; Mozart writes in a narrower range.")
    
    # Check 4: Bach should have high pitch class entropy (uses all 12 notes)
    if 'pitch_class_entropy' in means.columns:
        bach_pce = means.loc['bach', 'pitch_class_entropy']
        result = "PASS" if bach_pce > 0.75 else "INVESTIGATE"
        checks.append(f"  [{result}] Pitch class entropy: Bach = {bach_pce:.3f} (expected > 0.75)")
        checks.append(f"         Bach's fugues are highly chromatic and use all 12 pitch classes.")
    
    # Check 5: Chopin should have lower leap ratio (more stepwise)
    if 'leap_ratio' in means.columns:
        chopin_lr = means.loc['chopin', 'leap_ratio']
        beethoven_lr = means.loc['beethoven', 'leap_ratio']
        result = "PASS" if chopin_lr < beethoven_lr else "INVESTIGATE"
        checks.append(f"  [{result}] Leap ratio: Chopin ({chopin_lr:.3f}) < Beethoven ({beethoven_lr:.3f})")
        checks.append(f"         Chopin tends toward smooth stepwise motion; Beethoven uses dramatic leaps.")
    
    # Check 6: Debussy should have higher dissonance ratio
    if 'dissonance_ratio' in means.columns:
        debussy_dr = means.loc['debussy', 'dissonance_ratio']
        mozart_dr = means.loc['mozart', 'dissonance_ratio']
        result = "PASS" if debussy_dr > mozart_dr else "INVESTIGATE"
        checks.append(f"  [{result}] Dissonance ratio: Debussy ({debussy_dr:.3f}) > Mozart ({mozart_dr:.3f})")
        checks.append(f"         Debussy uses unresolved dissonances; Mozart resolves them classically.")
    
    for check in checks:
        print(check)
    
    passed = sum(1 for c in checks if '[PASS]' in c)
    total = sum(1 for c in checks if c.startswith('  ['))
    print(f"\n  Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("  Your features capture real musical differences. Ready for classification!")
    else:
        print("  Some checks didn't pass — investigate those features.")
        print("  (This doesn't necessarily mean something is wrong — musical")
        print("   intuition doesn't always match statistical averages.)")


def main():
    # Load data
    df, feature_cols = load_data()
    
    # Quality check
    check_data_quality(df, feature_cols)
    
    # Musical sanity check
    musical_sanity_check(df)
    
    # Generate all plots
    print("\n\n── GENERATING PLOTS ───────────────────────────────────")
    plot_histograms(df, feature_cols)
    plot_boxplots(df, feature_cols)
    plot_correlation_matrix(df, feature_cols)
    plot_radar_preview(df, feature_cols)
    
    print("\n\n" + "=" * 60)
    print("  EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"\n  All plots saved to {PLOT_DIR}/")
    print(f"  Open them and look for:")
    print(f"    1. Features where composers clearly separate (good for classifier)")
    print(f"    2. Features that look flat across all composers (may not be useful)")
    print(f"    3. Highly correlated features (might be redundant)")
    print(f"    4. Anything that contradicts musical intuition (possible bug)")
    print(f"\n  Next step: python notebooks/02_classifier.py")


if __name__ == '__main__':
    main()