# Musical DNA: Computational Fingerprinting of Compositional Style & Copyright Law

An interdisciplinary research project combining machine learning, music theory, and legal analysis to computationally identify what makes each composer's music unique — then applying those tools to analyze real music copyright disputes.

## Project Overview

**Component A — Style Fingerprinting Engine:** A classifier trained on MIDI data from 5–6 classical composers that extracts measurable "style signatures" using 25 musical features across melodic, harmonic, rhythmic, and structural categories.

**Component B — Copyright Case Predictor:** A hand-curated dataset of 60+ real music plagiarism cases, enriched with musical similarity metrics, used to predict whether a copyright case succeeds or fails.

## Repository Structure

```
musical-dna/
├── data/
│   ├── midi/              # MIDI files organized by composer
│   │   ├── bach/
│   │   ├── mozart/
│   │   ├── beethoven/
│   │   ├── chopin/
│   │   ├── debussy/
│   │   └── rachmaninoff/
│   ├── processed/         # Extracted feature matrices (CSV)
│   └── copyright_cases/   # Case dataset
├── src/
│   ├── features.py        # 25-feature extraction pipeline
│   ├── classifier.py      # Composer classification models
│   ├── similarity.py      # Pairwise similarity scoring
│   └── utils.py           # Shared helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_classifier.ipynb
│   └── 04_copyright_analysis.ipynb
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── tests/
│   └── test_features.py   # Feature extraction validation
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/musical-dna.git
cd musical-dna

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run feature extraction test
python tests/test_features.py
```

## Technology Stack

- **Python 3.11+** — core language
- **music21** — MIDI parsing and music analysis (MIT)
- **scikit-learn** — ML classifiers and evaluation
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **Streamlit** — interactive dashboard

## Research Questions

1. Can ML reliably distinguish composers using only computational features from MIDI scores?
2. Which musical features are most predictive of individual compositional style?
3. Can we predict music copyright case outcomes using quantitative musical features?
4. Do court rulings correlate with measurable similarity — or with non-musical factors?

## Author

**Nihal** — Rising Senior, Summer 2026

## License

MIT License — see [LICENSE](LICENSE) for details.
