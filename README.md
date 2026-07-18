# Musical DNA: Computational Fingerprinting of Compositional Style & Copyright Law

An interdisciplinary research project combining machine learning, music theory, and legal analysis. Component A computationally identifies what makes each classical composer's style unique. Component B applies those same tools to real music copyright disputes, asking whether courts' rulings track measurable musical similarity or something else entirely.

**🎛️ Live demo:** [musical-dna.streamlit.app](https://musical-dna.streamlit.app) — upload a MIDI to classify its composer, compare two works for similarity, and explore the copyright-case dataset.

**📄 Preprint:** [doi.org/10.5281/zenodo.21433890](https://doi.org/10.5281/zenodo.21433890) (CC BY 4.0)

The full writeup (methodology, results, five detailed case studies, and limitations) is in [`paper/musical_dna_draft.pdf`](paper/musical_dna_draft.pdf). A general-audience summary is in [`blog/musical_dna_blog_post.md`](blog/musical_dna_blog_post.md).

## Project Overview

**Component A, Style Fingerprinting Engine:** A Random Forest classifier trained on 353 MIDI files from six classical composers (Bach, Beethoven, Chopin, Debussy, Mozart, Rachmaninoff), extracting 21 features across melodic, harmonic, rhythmic, and structural dimensions. It reaches **88.7% test accuracy** on six-composer classification (an SVM edges it at 91.5%, but the Random Forest is kept as the primary model because it provides the feature importances the analysis and dashboard rely on). Applying the classifier to AI-generated music revealed a genre-bias limitation: a Carnatic classical piece (an Indian classical tradition unrelated to European composition) was confidently misclassified as Bach.

**Component B, Copyright Case Predictor:** A hand-curated dataset of 43 real music copyright cases (1946–2023), 33 of which are scored with computed melodic/harmonic/rhythmic/n-gram similarity between the plaintiff's and defendant's work. A logistic regression model finds that **non-musical case factors (plaintiff fame, defendant commercial success, expert testimony, litigation forum) predict outcomes far better than musical similarity alone** (AUC 0.86 vs. 0.39 under cross-validation), with plaintiff fame the single strongest predictor. Five outlier cases are analyzed in depth, including a synthesis where Component A's classical-composer engine is applied directly to disputed pop/rock recordings as an independent check.

## Repository Structure

```
musical-dna/
├── data/
│   ├── midi/                  # MIDI files (gitignored: composer + copyright-case audio)
│   ├── processed/              # Extracted feature matrices (CSV, gitignored)
│   └── copyright_cases/
│       └── cases.csv           # The 43-case copyright dataset
├── src/
│   ├── features.py             # 21-feature extraction pipeline (Component A)
│   ├── classifier.py           # Composer classifier training/evaluation
│   ├── similarity.py           # Pairwise melodic/harmonic/rhythmic/n-gram similarity (Component B)
│   ├── cases.py                 # Copyright case CSV loading, validation, auto-scoring
│   ├── case_model.py            # Logistic regression for case-outcome prediction
│   ├── batch_extract.py         # Batch feature extraction over data/midi/
│   ├── setup_data.py            # Bootstraps composer MIDI data from music21's corpus
│   └── import_maestro.py        # Alternative importer (MAESTRO piano dataset)
├── scripts/                     # Copyright-case audio pipeline (download → MIDI → score)
│   ├── download_audio.py
│   ├── convert_to_midi.py
│   └── link_midis.py
├── notebooks/                    # Weekly analysis scripts (plain .py, not Jupyter)
│   ├── 01_explore.py             # Sanity checks + exploratory plots
│   ├── 02_classifier.py          # Composer classifier training (Week 3)
│   ├── 03_style_maps.py          # Radar charts, t-SNE/PCA (Week 4)
│   ├── 04_ai_music_test.py       # AI-generated music generalization test
│   ├── 05_score_pair.py          # Auto-scores copyright case pairs
│   ├── 06_model_cases.py         # Logistic regression model (Week 6)
│   └── 07_outlier_analysis.py    # Outlier case studies + Component A/B synthesis (Week 7)
├── dashboard/                    # Streamlit app, 4 interactive tabs
│   ├── app.py
│   ├── tab_style_explorer.py     # Composer radar chart + feature breakdown
│   ├── tab_classifier.py         # Upload a MIDI file, get a predicted composer
│   ├── tab_case_explorer.py      # Filterable table of all 43 copyright cases
│   └── tab_similarity.py         # Upload two MIDI files, get a similarity score
├── tests/
│   └── test_features.py          # End-to-end pipeline validation (music21 built-in corpus)
├── paper/
│   ├── musical_dna_draft.md      # Full research paper (source)
│   └── musical_dna_draft.pdf     # Rendered PDF (~22 pages)
├── blog/
│   └── musical_dna_blog_post.md  # General-audience summary
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/NihalG51/musical-dna.git
cd musical-dna

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify the environment
python tests/test_features.py

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

Note: `data/midi/` (composer and copyright-case audio) and `data/processed/*.csv` are gitignored due to size and, for the copyright-case audio, copyright considerations, so they aren't included in this repository. See `src/setup_data.py` and `scripts/download_audio.py` for how to (re)build them.

## Technology Stack

- **Python 3.11+**, core language
- **music21**, MIDI parsing and music analysis
- **scikit-learn**, classifiers, logistic regression, cross-validation
- **pandas / numpy / scipy**, data manipulation and similarity metrics
- **matplotlib / seaborn**, visualization
- **Streamlit**, interactive dashboard
- **Basic Pitch** (Spotify), neural audio-to-MIDI transcription for copyright-case recordings
- **yt-dlp**, audio sourcing for copyright-case recordings

## Research Questions & Findings

1. **Can ML reliably distinguish composers using only computational features from MIDI scores?** Yes, at 88.7% accuracy across six composers, with pitch range, key stability, and average pitch the most predictive features.
2. **Does this generalize beyond the training distribution?** Not fully. An AI-generated Carnatic classical piece was confidently misclassified as Bach, revealing that the model learned "European art music" signatures rather than composer-specific ones.
3. **Can we predict music copyright case outcomes using quantitative musical features?** Weakly on their own (AUC 0.39, worse than chance), but combined with non-musical case facts, prediction improves substantially (AUC 0.86 from non-musical facts alone).
4. **Do court rulings correlate with measurable similarity, or with non-musical factors?** Non-musical factors, plaintiff fame most of all, are the stronger predictor in this dataset. The clearest illustration is *Williams v. Gaye* ("Blurred Lines"), which has the lowest computed similarity of any case examined, by two independent measures, despite being the highest-profile infringement verdict in the dataset.

## Author

**Nihal Gundluru**, Rising Senior, Summer 2026

## Citation

> Gundluru, N. (2026). *Musical DNA: Computational Fingerprinting of Compositional Style and Its Application to Music Copyright Analysis*. Zenodo. https://doi.org/10.5281/zenodo.21433890

## License

MIT License. See [LICENSE](LICENSE) for details.
