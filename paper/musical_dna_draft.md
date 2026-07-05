# Musical DNA: Computational Fingerprinting of Compositional Style and Its Application to Music Copyright Analysis

**Nihal**  
Independent Research Project, Summer 2026

---

## Abstract

This paper presents Musical DNA, a two-component computational system connecting music theory, machine learning, and copyright law. Component A is a style-fingerprinting engine trained on 353 MIDI files from six classical composers (Bach, Beethoven, Chopin, Debussy, Mozart, Rachmaninoff), extracting 21 musical features across melodic, harmonic, rhythmic, and structural dimensions. A Random Forest classifier reaches 88.7% test accuracy on six-composer classification, with an SVM marginally higher at 91.5%; its most discriminative features are pitch range, key stability, and average pitch. A generalization test produced one unexpected result: an AI-generated Carnatic classical vocal piece was confidently classified as Bach, exposing a genre-bias limitation relevant to evaluating AI-generated music. Component B applies pairwise similarity metrics to a hand-curated dataset of 43 real music copyright cases spanning 1946 to 2023, 33 of them fully scored with computed melodic, harmonic, rhythmic, and n-gram similarity. A logistic regression model trained on this data finds that non-musical factors (plaintiff fame, defendant commercial success, expert testimony, and litigation forum) predict case outcomes substantially better than musical similarity alone, with an AUC of 0.86 versus 0.39 under leave-one-out cross-validation and plaintiff fame the single strongest predictor. Five outlier cases, where computed similarity and the court's ruling diverge most, are examined as detailed studies, and Component A's general-purpose engine is applied directly to the disputed works as an independent check. One result stands out: *Williams v. Gaye* ("Blurred Lines") is among the least stylistically similar pairs in the outlier set despite its infringement verdict, while *Bright Tunes Music v. Harrisongs Music* ("My Sweet Lord") scores as highly similar under the general engine even where the specialized melodic metric is low. The paper contributes an open-source audio-to-MIDI similarity pipeline, a key-independent melodic comparison based on interval n-grams, a cross-validation methodology suited to a small case sample, and evidence that the legal standard of "substantial similarity" tracks non-musical case factors more closely than any computational similarity measure tested here.

---

## 1. Introduction

Music copyright law faces a measurement problem. Courts are asked to decide whether one song is "substantially similar" to another, but no agreed-upon standard exists for measuring musical similarity. In *Williams v. Gaye* (2018), a jury awarded $7.4 million in damages over songs that shared a rhythmic "feel" but not a single melodic note. In *Skidmore v. Led Zeppelin* (2020), a court reversed an earlier infringement verdict over a descending chromatic guitar figure that struck many listeners as obviously derivative. In *Gray v. Perry* (2020), a court found no infringement between two songs with near-identical harmonic structure, because the shared element (a repeating eight-note pattern) was deemed too simple to protect.

These rulings are legally inconsistent, and the stakes are rising. AI music generators like Suno and Udio now produce millions of tracks per month that may inadvertently reproduce copyrighted elements, and the legal system has no computational framework for evaluating such claims at scale.

This paper asks two questions. Can we quantify what courts mean by substantial similarity? And if so, do computational similarity scores predict court outcomes, or are legal rulings driven by factors beyond the music itself?

To answer them, I built two systems. The first, Component A, establishes that individual musical style can be captured computationally: a machine learning model distinguishes six classical composers with roughly 89% accuracy using only numerical features extracted from MIDI files. The second, Component B, applies the same tools to real copyright cases, measuring pairwise similarity between plaintiff and defendant works and comparing those scores against actual verdicts.

The synthesis shows that non-musical factors predict case outcomes considerably better than any computational similarity score I tested. It also identifies specific cases where computational similarity is high yet courts found no infringement, and, more surprisingly, cases where computational similarity is low yet infringement was found. Those cases expose the role of unprotectable common elements, access requirements, and factors entirely outside the music itself.

---

## 2. Background

### 2.1 Music Copyright Law

United States copyright law protects original musical works under 17 U.S.C. § 106. Infringement requires proof of (1) ownership of a valid copyright, (2) access to the work, and (3) substantial similarity between the works. The substantial similarity standard has no fixed definition and is applied inconsistently across circuits.

The foundational framework for similarity analysis was established in *Arnstein v. Porter* (2nd Cir. 1946), which split the inquiry into two stages: first, whether the defendant copied from the plaintiff (copying in fact); second, whether the copying was "improper appropriation." The second stage asks whether the defendant took protected expression rather than unprotectable ideas, common chord progressions, or musical building blocks.

Four doctrinal developments are relevant to this paper:

**Subconscious copying** (*Bright Tunes Music v. Harrisongs Music*, 1976): George Harrison was found liable for unconsciously reproducing the melodic hook of "He's So Fine" in "My Sweet Lord." The court held that intent is irrelevant. What matters is whether the defendant's work reproduces protected expression, regardless of how it got there.

**Thin copyright in simple elements** (*Gray v. Perry*, 2020): A repeating eight-note ostinato was found insufficiently original to protect, even though both songs used it in nearly identical form. The doctrine of "thin copyright" limits protection to highly original creative choices, not generic building blocks.

**Groove and feel protection** (*Williams v. Gaye*, 2018): A jury found infringement based on the overall groove and feel of "Got to Give It Up" and "Blurred Lines," despite minimal melodic overlap. Critics warned the ruling could make it dangerous to write music in the style of an earlier artist.

**De minimis sampling** (*VMG Salsoul v. Ciccone*, 9th Cir. 2016; *Bridgeport Music v. Dimension Films*, 6th Cir. 2005): The circuits are split on whether a tiny, inaudible sample constitutes infringement. The 6th Circuit says any sample requires a license; the 9th Circuit applies a de minimis threshold. The Supreme Court has not resolved the split.

### 2.2 Computational Music Analysis

Music Information Retrieval (MIR) is the field of extracting meaningful information from musical signals. Prior work includes composer identification via pattern recognition on low-level counterpoint features (Backer & van Kranenburg, 2005), melodic similarity (Müller, 2015), and rhythmic pattern recognition (Honing, 2013). The music21 library (Cuthbert & Ariza, 2010) provides a Python toolkit for symbolic music analysis that forms the technical backbone of this project.

Prior work applying computational methods to music copyright is closer in spirit to this paper's Component B than to Component A, and one study is a direct predecessor. Cronin (1998) was among the first to formalize concepts of melodic similarity specifically for copyright infringement analysis. More recently, Yuan, Cronin, Müllensiefen, Fujii, and Savage (2023) compared human perceptual judgments and automated similarity algorithms against actual court rulings across 40 real music copyright cases. They found that listeners evaluating full audio recordings (roughly 58% accuracy) outperformed both melody-only listening and the automated algorithms tested. That neither computation nor careful human listening reliably reproduces court outcomes anticipates this paper's own finding: non-musical case facts, not musical similarity of any kind, are the strongest quantitative predictor of outcome in my dataset (Section 5.4). Fishman (2018) separately noted that courts lack tools for distinguishing protectable from unprotectable musical elements. This paper builds on that work by separating melodic, harmonic, and rhythmic similarity into distinct measurable dimensions and testing each against outcome individually, rather than relying on a single composite judgment.

---

## 3. Dataset

### 3.1 Component A: Composer MIDI Dataset

The composer classification dataset consists of 353 MIDI files representing six composers from the Baroque through Late Romantic periods:

| Composer | Era | Pieces | Source |
|---|---|---|---|
| J.S. Bach | Baroque | 100 | music21 built-in corpus, KernScores |
| Beethoven | Classical–Romantic | 59 | music21 corpus, IMSLP |
| Chopin | Romantic | 50 | IMSLP, MuseScore |
| Debussy | Impressionist | 40 | IMSLP |
| Mozart | Classical | 54 | music21 corpus |
| Rachmaninoff | Late Romantic | 50 | IMSLP |

All files are solo piano or keyboard reductions. Orchestral MIDI files were excluded because inconsistent instrument mapping produces noisy features. Files that failed to parse or produced missing values on more than three features were discarded.

### 3.2 Component B: Copyright Case Dataset

The copyright case dataset is an original contribution of this project. It contains 43 hand-curated cases spanning 1946–2023, each coded with 16 fields including:

- **Case metadata**: name, year, outcome (binary: 0 = no infringement, 1 = infringement found)
- **Musical description**: plaintiff work, defendant work, genre, elements at issue (melody, harmony, rhythm, lyrics, groove)
- **Computed similarity**: melodic overlap score, harmonic overlap score, rhythmic overlap score (0–1 scale)
- **Non-musical factors**: plaintiff fame (1–5), defendant commercial success (1–5), expert testimony presence, jurisdiction, settlement status

The dataset contains 43 cases, 33 of which are fully scored with computed similarity metrics at the time of writing. The remaining 10 involve a contested element unsuited to MIDI-based melodic comparison, such as a 0.23-second drum or horn sample, a spoken-word sample, or a lyrics-only claim. These are kept for their factual and doctrinal value (for example, the *Bridgeport Music* sampling suits that established the 6th/9th Circuit split on de minimis sampling) but carry no similarity score.

Among the 33 scored cases, outcomes are reasonably balanced (19 infringement, 14 no infringement). Genres skew toward Pop (11) and Rock (10), with a substantial Hip-Hop contingent (6) and smaller R&B/Pop, Musical Theater, and R&B representation. Jurisdiction is highly heterogeneous: 18 distinct values across 33 rows, spanning U.S. circuit and district courts, the U.S. Supreme Court, UK courts, and the Federal Court of Australia, plus several disputes that never reached a court and were resolved by pre-suit settlement. Because that heterogeneity makes jurisdiction impossible to one-hot encode usefully at this sample size (Section 4.4), it is collapsed to a single binary indicator: whether the case proceeded through formal U.S. federal litigation.

All melodic, harmonic, rhythmic, and n-gram scores in the dataset are computed algorithmically by the pairwise similarity engine (Section 4.3), not scored by ear. An earlier plan to hand-score cases by listening was dropped once the interval-based engine could run end-to-end on found audio. Audio for scored cases came from YouTube via yt-dlp, or in a few cases (for example, *Arnstein v. Porter* and *Stratchborneo v. Arc Music*) from archival recordings hosted by the Music Copyright Infringement Resource (MCIR) at George Washington University Law School, where the disputed work is too obscure to appear on commercial streaming platforms. Conversion to MIDI used Spotify's Basic Pitch model (Bittner et al., 2022), a neural audio-to-MIDI transcriber, with audio trimmed to the first 90 seconds to focus on the contested passage and reduce processing time. Section 6.3 discusses that trimming as a limitation.

---

## 4. Methodology

### 4.1 Feature Extraction (Component A)

Each MIDI file is processed with music21 to extract 21 scalar features grouped into four categories. Every feature function is wrapped in a safe extractor that returns 0 on failure, so a single corrupt file does not drop the entire row.

**Melodic features (6):** pitch range (highest minus lowest MIDI note), average pitch, pitch class entropy (Shannon entropy of the 12-note pitch class distribution), leap ratio (proportion of melodic intervals larger than a major second), melodic contour (ratio of ascending to descending motion), and interval-histogram size (the count of distinct melodic interval types used).

**Harmonic features (5):** key stability (Krumhansl-Schmuckler key-finding correlation score), modulation frequency (key changes per 16-bar window), chord vocabulary size (number of distinct chord types), dissonance ratio (proportion of minor-second and tritone intervals), and tonal gravity (density of V→I and viio→I cadential resolutions).

**Rhythmic features (5):** note density (notes per quarter-note beat), duration variance, syncopation index (proportion of notes on weak metric positions), rest ratio, and rhythmic entropy (Shannon entropy of the duration histogram).

**Structural features (5):** piece length in measures, repetition ratio (cosine similarity between consecutive 4-bar feature windows), dynamic range (MIDI velocity maximum minus minimum), voice count, and texture density (mean simultaneous notes per beat position).

### 4.2 Classifier Training (Component A)

The 353-piece dataset was split into 80% training and 20% test sets using stratified sampling to preserve class proportions. Three classifiers were trained and compared using 5-fold stratified cross-validation:

1. **Random Forest** (200 trees, max depth unlimited)
2. **SVM** (RBF kernel, C=10, gamma='scale')
3. **Gradient Boosting** (200 estimators, max depth 4, learning rate 0.1)

All three were wrapped in a scikit-learn pipeline with StandardScaler preprocessing. Feature importances were extracted from the Random Forest using mean decrease in impurity.

### 4.3 Pairwise Similarity Engine (Component B)

Four similarity metrics are computed for each plaintiff–defendant MIDI pair:

**Melodic similarity** compares the melodic interval sequences of both works, meaning the sequence of semitone differences between consecutive notes rather than absolute pitch values. This design choice is important. Two songs with the same melody in different keys produce identical interval sequences and score as similar, whereas an absolute-pitch comparison would return near-zero similarity. The score is the SequenceMatcher ratio (2M/T, where M is the number of matching elements and T is the total elements across both sequences) computed on the interval representation.

**Harmonic cosine similarity** chordifies each score using music21's built-in harmonic reduction, builds a frequency histogram of chord types (for example, "major triad" and "minor seventh chord"), and computes the cosine similarity of the resulting vectors. This metric is key-independent by design and captures shared harmonic vocabulary regardless of voicing or transposition.

**Rhythmic DTW similarity** applies dynamic time warping to the inter-onset interval (IOI) sequences of both works, meaning the time gaps between consecutive note onsets, measured in quarter-note beats. DTW allows flexible alignment of rhythmic patterns that may be stretched or compressed, following the general approach used for audio-to-MIDI sequence alignment in Raffel (2016). The raw DTW distance is normalized by `avg_sequence_length × mean_IOI`, converting it to a similarity score in [0, 1].

**N-gram overlap** computes the Jaccard index of 4-interval melodic n-grams, meaning all length-4 subsequences of the interval sequence. This captures shared short melodic phrases regardless of their position in the song, and it is more robust to polyphonic transcription noise than the sequential match used in melodic similarity.

**Melody extraction from polyphonic audio.** Because Basic Pitch transcribes full-band audio into a single-track MIDI containing all audible pitches, a melody proxy is required. The top 30% of notes by pitch value are kept as the melody representation, since in a full-band recording the melodic line tends to sit above the accompaniment in pitch space. Scores are evenly subsampled to 200 notes before comparison so that dense files do not dominate the sequence match.

### 4.4 Logistic Regression for Case Outcome (Component B)

To directly test whether musical similarity predicts case outcomes better than non-musical case factors, three logistic regression models were fit and compared:

1. **Musical only**: melodic, harmonic, rhythmic, and n-gram overlap scores (4 features)
2. **Non-musical only**: plaintiff fame, defendant commercial success, expert-testimony presence, and a binary indicator for whether the case proceeded through formal U.S. federal litigation (4 features)
3. **Combined**: all 8 features above

All features were standardized (zero mean, unit variance) before fitting an L2-regularized logistic regression (scikit-learn default `C=1.0`). Regularization guards against the class-separation instability that unregularized logistic regression is prone to when the number of features is a meaningful fraction of the sample size.

With only 33 scored cases, a conventional 80/20 train/test split would leave roughly 7 held-out cases, too few for a trustworthy single accuracy estimate. Evaluation therefore relies on two cross-validation strategies applied to the whole dataset: stratified 5-fold cross-validation (reporting mean accuracy, precision, recall, and AUC-ROC across folds), and leave-one-out cross-validation (fitting on 32 cases and predicting the 33rd, repeated for every case), which trades some optimism for much lower variance on a dataset this small.

One candidate feature, whether the case was resolved by settlement, was deliberately excluded. Inspection revealed it was not a genuine predictor but a restatement of the label: every settled case in the dataset has its outcome coded as 1 ("infringement implied") by definition at data-entry time, since a settlement produces no independent court ruling to measure against. `settled = 1` implies `outcome = 1` with perfect consistency across all 13 settled cases. Including it is a textbook case of target leakage. It inflated cross-validated AUC to 0.97 in an earlier version of this analysis, a number that should not be read as a real finding, and it was removed before the final results were computed.

### 4.5 Outlier Identification and Component A/B Synthesis

Two further analyses probe where computational similarity and legal outcome diverge most sharply, and whether a second, independently-built similarity measure agrees with the purpose-built one.

**Outlier identification.** A composite musical similarity score (the mean of melodic, harmonic, and rhythmic overlap) is computed for each case. Cases are ranked within two pools: no-infringement rulings ranked by highest composite similarity (courts finding no infringement despite apparent similarity), and infringement rulings, restricted to cases that reached an actual verdict rather than a settlement, ranked by lowest composite similarity (courts finding infringement despite apparent dissimilarity). Cases with an exact-zero rhythmic score are excluded from both pools. A score of exactly 0.000 is a known artifact of the 90-second trim window missing the contested passage entirely (documented for *Williams v. Broadus* and *Larrikin Music Publishing v. EMI Songs Australia* in the dataset), not a genuine finding of zero rhythmic similarity, so using it as a "low similarity" exemplar would be misleading.

**Component A/B synthesis.** For the resulting five outlier cases, Component A's 21-feature style-fingerprinting engine, built to distinguish six classical composers rather than to compare two arbitrary pop or rock recordings, is applied directly to the plaintiff and defendant MIDI files. Each piece's 21-feature vector is standardized against the classical-composer training distribution (the same distribution the composer classifier was trained on), and cosine similarity is computed between the two standardized vectors. Because this engine was never designed with copyright disputes in mind, it produces a similarity measure conceptually independent of Component B's purpose-built melodic, harmonic, and rhythmic engine, which makes agreement or disagreement between the two measures informative on its own.

---

## 5. Results

### 5.1 Composer Classification

| Model | CV Accuracy (5-fold) | Test Accuracy |
|---|---|---|
| Random Forest | 84.4% ± 4.0% | **88.7%** |
| SVM (RBF) | 84.0% ± 2.5% | 91.5% |
| Gradient Boosting | 81.2% ± 4.1% | 87.3% |

All three classifiers cluster within a few points of one another. The SVM has the single highest test accuracy (91.5%), but I keep the Random Forest as the primary model throughout this paper for two reasons. Its cross-validation accuracy (84.4%) is statistically indistinguishable from the SVM's (84.0%), and it is the only one of the three that yields the feature importances the next section relies on and that the interactive dashboard's style explorer is built around. The Random Forest's test accuracy is 88.7%, comfortably above the 70% project target. The gap between its cross-validation accuracy (84.4%) and test accuracy (88.7%) shows how sensitive a single 71-piece test split is at this dataset size. A few pieces moving between correct and incorrect shifts the headline number by several points, which is why the cross-validation estimate is the more trustworthy figure.

Per-composer breakdown on the test set (Random Forest):

| Composer | Precision | Recall | F1 | Test Pieces |
|---|---|---|---|---|
| Bach | 100.0% | 95.0% | 97.4% | 20 |
| Beethoven | 75.0% | 100.0% | 85.7% | 12 |
| Chopin | 80.0% | 80.0% | 80.0% | 10 |
| Debussy | 100.0% | 62.5% | 76.9% | 8 |
| Mozart | 100.0% | 90.9% | 95.2% | 11 |
| Rachmaninoff | 81.8% | 90.0% | 85.7% | 10 |

Bach achieves the highest F1 (97.4%), consistent with his distinctive contrapuntal texture. Debussy again has the lowest recall (62.5%): three of its eight test pieces are misclassified (two as Chopin, one as Rachmaninoff), likely because its tonal ambiguity, whole-tone harmonies, and sparse textures produce feature vectors that overlap with the late-Romantic composers. The other notable pattern is Beethoven's lower precision (75.0%). It attracts a scatter of false positives, with one piece each from Bach, Chopin, Mozart, and Rachmaninoff misclassified as Beethoven, reflecting how stylistically central and wide-ranging his writing is as it sits between the Classical and Romantic clusters.

![Confusion matrix for the Random Forest on the 71-piece test set (88.7% accuracy). Debussy is the weakest class (three of eight pieces misclassified, two as Chopin), and Beethoven attracts several single-piece false positives from other composers, consistent with the per-composer precision and recall above.](../notebooks/plots/05_confusion_matrix.png){width=65%}

Figure 2 shows the same separability directly. A t-SNE projection of all 353 pieces, colored by composer with an X marking each composer's centroid, shows Bach and Mozart forming tight, well-separated clusters on the left. The three Romantic and Impressionist composers, Chopin, Debussy, and Rachmaninoff, have centroids that sit close together on the right and overlap more, mirroring the confusion matrix's error pattern.

![t-SNE projection (perplexity=30) of the full 353-piece dataset. Bach's contrapuntal keyboard writing and Mozart's Classical clarity form the most isolated clusters; the three Romantic-era composers cluster more closely together, foreshadowing where the classifier's few errors occur.](../notebooks/plots/08_tsne.png){width=85%}

### 5.2 Feature Importance Analysis

The ten most predictive features from the Random Forest, ranked by mean decrease in impurity:

| Rank | Feature | Importance | Musical Interpretation |
|---|---|---|---|
| 1 | pitch_range | 0.155 | Bach's counterpoint spans wide ranges; Debussy compresses melody |
| 2 | key_stability | 0.072 | Debussy's tonal ambiguity vs. Bach's tonal certainty |
| 3 | average_pitch | 0.069 | Distinguishes bass-heavy from treble-dominant composers |
| 4 | note_density | 0.065 | Rachmaninoff's dense textures vs. Mozart's clarity |
| 5 | leap_ratio | 0.064 | Bach's angular lines vs. Chopin's smooth cantabile |
| 6 | dissonance_ratio | 0.057 | Debussy's chromaticism vs. Bach's consonant counterpoint |
| 7 | modulation_frequency | 0.054 | Beethoven's dramatic key shifts vs. Mozart's stability |
| 8 | piece_length | 0.054 | Bach's shorter keyboard pieces vs. Romantic extended forms |
| 9 | dynamic_range | 0.044 | Romantic dynamic contrast vs. Baroque terraced dynamics |
| 10 | chord_vocabulary_size | 0.042 | Rachmaninoff's rich chord palette |

These results align with established music theory. Pitch range ranks first because Bach's counterpoint spans wide ranges across multiple voices, in contrast to the more compressed melodic ranges of the Romantics. Key stability ranking second matches the long-held observation that Debussy's impressionist language deliberately undermines tonal certainty. The corrected `tonal_gravity` feature, which measures cadential pull toward the tonic, is now a live signal that separates densely cadential Baroque writing from Debussy's ambiguity, but it lands outside the top ten and contributes modestly rather than dominating.

![Top 10 features ranked by mean decrease in impurity. Pitch range alone accounts for more than twice the importance of the second-ranked feature (key stability), making it by a wide margin the single most useful measurement for distinguishing these six composers.](../notebooks/plots/06_feature_importance.png){width=85%}

### 5.3 AI Music Generalization Test

To test whether the classifier generalizes beyond its training distribution, I applied it to ten tracks generated by AI tools (Suno and Udio) in the style of the six composers. Nine of the ten were classified correctly. The exception was telling: a Carnatic classical vocal piece, an Indian classical tradition with no direct relation to European composition, was confidently classified as **Bach** with high probability.

The result reveals a genre bias. The classifier has learned the style signatures of European art music, and when it met Carnatic music, which shares some surface features with Baroque counterpoint such as complex ornamentation, modal scales, and systematic rhythmic patterns, it defaulted to the nearest label it had. This matters for any use of these tools on culturally diverse music, and it is a reminder that "style fingerprinting" is always relative to the training distribution.

### 5.4 Copyright Case Outcome Model

Table 5.4a compares the three logistic regression models described in Section 4.4, evaluated via stratified 5-fold cross-validation and leave-one-out cross-validation (LOO) across all 33 scored cases.

**Table 5.4a. Feature set comparison**

| Feature set | Accuracy (5-fold) | AUC-ROC (5-fold) | LOO Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| Musical only (4 features) | 51.0% ± 12.9% | 0.39 ± 0.07 | 36.4% | 55.3% | 80.0% |
| Non-musical only (4 features) | 75.7% ± 7.4% | **0.86 ± 0.10** | **72.7%** | 84.3% | 73.3% |
| Combined (8 features) | 70.0% ± 8.5% | 0.68 ± 0.10 | 63.6% | 76.3% | 73.3% |

The headline result is clear. Non-musical factors alone predict case outcomes far better than musical similarity alone (AUC 0.86 versus 0.39). The musical-only AUC of 0.39 sits below the 0.50 chance baseline, meaning that on this sample the musical-only model's ranking of cases is, if anything, mildly anti-correlated with outcome. Adding musical similarity to the non-musical model does not help; it hurts, with AUC dropping from 0.86 to 0.68. That is consistent with the musical features adding more estimation noise than signal once the stronger non-musical predictors are already present, which is what one expects when fitting an 8-parameter model to 33 rows.

**Table 5.4b. Full-model standardized coefficients** (fit on all 33 cases; odds ratio > 1 pushes toward "infringement found")

| Feature | Coefficient | Odds Ratio |
|---|---|---|
| plaintiff_fame | **+0.972** | 2.64 |
| expert_testimony | −0.805 | 0.45 |
| defendant_commercial_success | +0.804 | 2.23 |
| us_court | −0.628 | 0.53 |
| melodic_overlap_score | +0.228 | 1.26 |
| rhythmic_overlap_score | +0.187 | 1.21 |
| ngram_overlap | −0.070 | 0.93 |
| harmonic_overlap_score | −0.008 | 0.99 |

Plaintiff fame is the single strongest predictor in the combined model. A one-standard-deviation increase in plaintiff fame is associated with 2.64 times higher odds of an infringement finding, holding everything else constant. Melodic similarity's coefficient (+0.228) is less than a quarter the size of plaintiff fame's, and all four musical-similarity coefficients are small relative to the two strongest non-musical ones.

![Standardized coefficients from the combined (8-feature) logistic regression, fit on all 33 scored cases. The two largest-magnitude coefficients (plaintiff_fame, expert_testimony) are both non-musical; all four musical-similarity coefficients (bottom of the chart) are comparatively small.](../notebooks/plots/07_case_model_coefficients.png){width=80%}

**A note on a leakage artifact.** An earlier version of this analysis included settlement status as a fifth non-musical feature and produced an AUC of 0.97, a number that would have made a dramatic headline. It does not appear in Table 5.4a because it was leakage rather than a finding. Every settled case in this dataset has its outcome coded as 1 by definition, since a settlement produces no independent ruling to measure the musical merits against. The feature was removed once this was clear (Section 4.4), and the corrected numbers above are the ones to cite from this analysis.

### 5.5 Outlier Case Studies

Following the procedure in Section 4.5, five cases were selected as the sharpest divergences between computed musical similarity and the actual court ruling. Three are cases where similarity was unusually high yet no infringement was found. Two, restricted to cases with a real trial verdict rather than a settlement, are cases where similarity was unusually low yet infringement was found.

**Swirsky v. Carey** (9th Cir. 2004) pits Sandra Swirsky and Henry Marsh's "One of Those Love Songs" against Mariah Carey's "Always Be My Baby." Its composite similarity of 0.601 (melodic 0.37, harmonic 0.95, rhythmic 0.48) is the highest of any no-infringement case in the 33-case dataset. The district court granted summary judgment for Carey, but the Ninth Circuit reversed. Its reasoning: when qualified musicologists reach conflicting conclusions about whether a combination of individually unprotectable elements (here, an eight-note pitch sequence combined with a particular rhythm) is substantially similar, that disagreement is itself evidence a jury must weigh, not a question a court can resolve on the papers. At the eventual trial, the jury found no infringement.

The case illustrates a distinction the similarity engine cannot represent: procedural posture is not the same as outcome. A reversal of summary judgment means only that the case was close enough to require a trial, not that the reversing court believed infringement occurred. The composite score of 0.601, the highest of any no-infringement case, fits the Ninth Circuit's own assessment that this was a close call worth putting to a jury. It was the jury, not the algorithm, that drew the final line between "close" and "infringing."

**Repp v. Webber** (S.D.N.Y. 1994; 2d Cir. 1997) shows the same procedural pattern three years earlier and an ocean away in genre: Ray Repp's 1978 liturgical song "Till You" against Andrew Lloyd Webber's "Phantom Song" from *The Phantom of the Opera*. Composite similarity is 0.562, driven almost entirely by a harmonic score of 0.946, one of the highest in the whole dataset. As in *Swirsky*, the district court's summary judgment for the defendant was reversed on appeal because conflicting expert testimony on melodic contour and harmonic idiom created a triable issue of fact. As in *Swirsky*, the case went to a jury, and the jury (in December 1998) found no infringement. Webber's own counterclaim, that Repp's song had copied his earlier "Close Every Door," was separately dismissed for insufficient similarity. Together these outcomes show that near-identical harmonic vocabulary between two tonal, diatonic pop songs is common enough that courts and juries are reluctant to treat it alone as proof of copying.

**Sheeran v. Chokri** ([2022] EWHC 827 (Ch)) is the clearest case in this dataset of access, not similarity, deciding the outcome. Sami Chokri's "Oh Why" and Ed Sheeran's "Shape of You" share a composite similarity of 0.532 under Component B's engine and a high Component A style-fingerprint similarity of 0.797 (see Section 5.6), among the closest pairings examined on both measures. Both an independently built general style engine and a purpose-built melodic, harmonic, and rhythmic engine agree that the two songs occupy unusually similar musical territory. Yet the UK High Court granted Sheeran a declaration of non-infringement, because Chokri could not show Sheeran had ever heard "Oh Why." It had aired on radio only twice and had roughly 13,000 YouTube views when "Shape of You" was written. Justice Zacaroli found that Sheeran had "neither deliberately nor subconsciously" copied the phrase. The case is direct evidence that no similarity metric, however sophisticated, and however much two independent tools agree, can substitute for proof of access, one of the three elements of the *Arnstein* test that sits entirely outside what any acoustic measurement can capture.

**Bright Tunes Music v. Harrisongs Music** (S.D.N.Y. 1976), the case in which George Harrison's "My Sweet Lord" was found to have subconsciously copied The Chiffons' "He's So Fine," has the lowest composite similarity (0.362) of any infringement case that reached a real verdict in this dataset, driven by a low melodic score (0.140). Taken alone, Component B's engine would rate this one of the weaker melodic matches examined. But Component A's independent style-fingerprint comparison (Section 5.6) finds the two pieces highly similar overall (cosine similarity 0.826, among the highest of any case examined), and the harmonic score (0.729) is comparatively strong too. One plausible explanation is that the specific four-note hook central to the court's finding is diluted within the full 90-second melodic sequence used for the sequence match, while broader stylistic features such as key stability, rhythmic profile, and note density, which the general engine captures, reflect a closer overall resemblance. The case is a reminder that "subconscious copying" asks about a holistic impression rather than a single quantified score, and that different computational lenses on the same pair of recordings can tell different parts of that story.

**Williams v. Gaye** (9th Cir. 2018) is the most consequential outlier in the dataset, and the case where the two independent tools built for this project most agree with each other and most disagree with the jury. Marvin Gaye's "Got to Give It Up" and Robin Thicke, Pharrell Williams, and T.I.'s "Blurred Lines" register only a moderate composite similarity (0.399: melodic 0.236, harmonic 0.599, rhythmic 0.362) under Component B's specialized engine, with none of the three dimensions standing out relative to the rest of the dataset. Component A's general engine goes further. Trained only on classical piano repertoire, with no awareness of this case or even this genre, it rates these two recordings the least stylistically similar pair in the five-case outlier set (cosine similarity 0.511, well below *Swirsky*'s 0.827 or *Bright Tunes*' 0.826). Two tools built independently, for different purposes, neither designed with legal argument in mind, both find less objective similarity here than the 2015 jury's $7.4 million verdict implied. That fits the widespread musicological and legal criticism that the *Blurred Lines* verdict protected an unprotectable "feel," meaning groove, timbre, and production choices, rather than any specific borrowed melodic or harmonic expression. One caveat matters: neither tool was built to measure that kind of feel. Their agreement that the songs are not especially similar is evidence about melody, harmony, rhythm, and general compositional style, not about groove.

### 5.6 Component A/B Synthesis

Table 5.6 applies Component A's 21-feature style-fingerprinting engine, trained to distinguish six classical composers and used here without modification, directly to the plaintiff and defendant recordings in the five outlier cases. Each recording is standardized against the same classical-composer feature distribution the composer classifier uses (Section 4.5).

**Table 5.6. Independent style-fingerprint comparison**

| Case | Style-FP cosine sim. | Melodic | Harmonic | Rhythmic | Ruling |
|---|---|---|---|---|---|
| Swirsky v. Carey | 0.827 | 0.37 | 0.95 | 0.48 | No infringement |
| Bright Tunes v. Harrisongs | 0.826 | 0.14 | 0.73 | 0.22 | **Infringement** |
| Sheeran v. Chokri | 0.797 | 0.34 | 0.85 | 0.40 | No infringement |
| Repp v. Webber | 0.659 | 0.34 | 0.95 | 0.40 | No infringement |
| Williams v. Gaye | **0.511** | 0.24 | 0.60 | 0.36 | **Infringement** |

Four of the five cases cluster tightly between 0.66 and 0.83 on the general style-fingerprint measure, so the two tools broadly agree that those pairs occupy similar musical territory whatever the ruling. *Williams v. Gaye* stands apart at 0.511. It scores lowest on both the specialized composite and the general style-fingerprint measure, even though it is the only one of the five outliers with a straightforward jury verdict for the plaintiff. The most notable single-case divergence between the two tools is *Bright Tunes*, where the general engine registers high similarity (0.826) even though the specialized melodic score alone is low (0.14). The case with the least algorithmic support for similarity, by either measure, is also the case most publicly criticized as overextending copyright. That may be coincidence, but it is at least consistent with both independently built tools detecting something the *Blurred Lines* jury weighed differently.

---

## 6. Discussion

### 6.1 The Computational–Legal Gap

The most important finding here is not what the algorithm gets right, but what it cannot measure. The similarity scores capture acoustic similarity: how much two works share in pitch patterns, chord vocabulary, and rhythmic feel. Courts decide legal similarity: how much protected expression was taken, filtered through access, protectability, and factors that have nothing to do with the music. Section 5.4 makes that gap quantitative rather than merely observational. A model built entirely from non-musical case facts (plaintiff fame, defendant commercial success, expert testimony, and forum) predicts outcomes with an AUC of 0.86, while a model built entirely from musical similarity predicts with an AUC of 0.39, worse than a coin flip. If "substantial similarity" doctrine were mainly measuring what this engine measures, the musical-only model should be the stronger one. It is not, by a wide margin.

This does not mean musical similarity is irrelevant to how courts reason. The case studies in Section 5.5 show it is often central to the parties' arguments and the court's language. It means that among the cases in this dataset, similarity alone is a weak statistical predictor of which way the ruling went once access, fame, forum, and settlement dynamics are accounted for. Four cases make the shape of that gap concrete:

- **Gray v. Perry**: Harmonic similarity 0.923, one of the highest in the dataset, yet no infringement. The court agreed the songs were harmonically near-identical but ruled the shared element, a simple repeating eight-note ostinato, unprotectable under thin-copyright doctrine. Computational similarity is necessary but not sufficient for legal liability.

- **Sheeran v. Chokri**: Composite similarity 0.532 and a high style-fingerprint similarity (0.797), yet no infringement, because the court found no access. No similarity metric, however high, can substitute for proof that the defendant ever heard the plaintiff's work.

- **Bright Tunes v. Harrisongs**: A low melodic score (0.140) but high harmonic (0.729) and style-fingerprint (0.826) similarity, and a finding of subconscious copying. Different similarity lenses on the same recordings can disagree with each other even while agreeing with the court.

- **Williams v. Gaye**: The lowest style-fingerprint similarity (0.511) and an unremarkable composite score (0.399) of any outlier, yet the highest-profile infringement verdict in the dataset. This is the case where computational similarity, by two independently built measures, diverges most from the legal outcome, and it is also the case most publicly criticized as an overextension of copyright.

Taken together, these patterns refine the project's original hypothesis. An early, four-case reading suggested that rhythmic similarity might predict outcomes across the board, but the larger sample does not bear that out: in Table 5.4a, rhythmic similarity's coefficient is modest and musical similarity as a group performs worse than chance. The strongest quantitative predictors of outcome are the non-musical facts of who is suing whom, in what forum, and with what evidence. The most doctrinally interesting cases are the ones where musical similarity and those non-musical facts point in different directions.

### 6.2 What Makes a Composer Sound Like Themselves

The feature importance analysis offers specific, testable answers to the question "what makes Mozart sound like Mozart?"

- **Bach** is defined by wide pitch range (dense counterpoint spanning multiple octaves) and low dissonance ratio (meticulous voice-leading consonance)
- **Debussy** is defined by low key stability (deliberate tonal ambiguity) and high dissonance ratio (chromatic clusters, whole-tone scales)
- **Chopin** is defined by low leap ratio (smooth cantabile melody) and high note density (ornamental passage work)
- **Rachmaninoff** is defined by large chord vocabulary (chromatic late-Romantic harmony) and high note density (thick textures)
- **Mozart** is defined by high rest ratio (clear phrase endings) and moderate key stability (Classical tonal clarity with occasional dramatic modulations)

![All six composer "fingerprints" overlaid on the top 8 Random Forest features, each normalized 0–1 across the dataset. Debussy's extreme pitch_range and modulation_frequency values, and Bach and Chopin's shared extremes on key_stability and leap_ratio, are visible at a glance. These are the same features driving both the importance ranking (Figure 3) and the classifier's few confusions (Figure 1).](../notebooks/plots/07b_style_fingerprints_overlay.png){width=70%}

That these computational findings match what music theorists have long observed is reassuring. It suggests the features are capturing real musical structure rather than statistical artifacts of the dataset.

### 6.3 Limitations

**Polyphonic audio transcription.** Basic Pitch was designed for single-instrument audio. Applied to full-band recordings (drums, bass, guitars, synths, vocals), it produces noisy single-track MIDI containing all detected pitches. The top-30% pitch proxy for melody extraction is an approximation that works in practice but is not theoretically grounded. Accuracy would improve substantially with access to isolated stem recordings.

**90-second trimming.** Audio is trimmed to the first 90 seconds before transcription to reduce processing time. This focuses the comparison on the intro and first chorus, usually the contested passage, but it misses cases where the similarity appears in a bridge or later section.

**Key-independent comparison.** Interval-based melodic comparison handles transposition but not inversion, retrograde, or augmentation. A composer who takes a melody and plays it backwards, upside-down, or twice as slowly would score 0 on our melodic metric even if the borrowing is legally actionable.

**Small scored sample and its modeling consequences.** 33 scored cases is a small sample for logistic regression, and this shapes what the model in Section 5.4 can and cannot claim. An 8-feature combined model fit on 33 rows is close to the point where the rule-of-thumb "10 events per predictor" heuristic starts to strain. That is why evaluation relies on cross-validation (5-fold and leave-one-out) rather than a single held-out test set, and why the paper reports comparative AUC across feature sets rather than treating any one model's accuracy as a precise, generalizable number. The result that non-musical factors outpredict musical similarity is large enough (0.86 versus 0.39 AUC) to be unlikely to be pure sampling noise, but the exact magnitude should be read as indicative rather than definitive until the dataset grows. A related caution: settlement status turned out to be a near-perfect restatement of the outcome label under this dataset's coding convention, and was excluded after producing a misleadingly high AUC of 0.97 in an earlier pass. With few rows and few features, a single leaky variable can dominate the result, so every feature added to a small-sample model should be checked for this kind of circularity before it is trusted.

**Rhythmic-score artifacts from the trim window.** Two cases (*Williams v. Broadus* and *Larrikin Music Publishing v. EMI Songs Australia*) score exactly 0.000 on rhythmic similarity, which the outlier-selection method (Section 4.5) treats as a measurement artifact rather than a real finding. The 90-second trim window most likely misses a contested passage that occurs later in the track. This is a genuine limitation of the current pipeline for hip-hop sampling cases, where the sampled riff is often not in the intro. It should be fixed by trimming around a manually identified timestamp rather than always the first 90 seconds before those two cases are used in any further quantitative analysis.

**Settled cases lack an independent ruling.** 13 of the 43 cases were resolved by settlement rather than trial or summary judgment. These cases add valuable factual and doctrinal coverage, but their "infringement implied" outcome coding is a judgment call about what a settlement suggests, not a court's actual finding. That is exactly why they were excluded from the "low similarity, infringement found" outlier pool in Section 4.5 and from any predictor in Section 5.4's models.

**Training distribution bias.** The classifier was trained only on European art music from 1600 to 1940. As the Carnatic finding shows, it does not generalize beyond that distribution. Copyright cases span hip-hop, pop, R&B, rock, and electronic music, genres with fundamentally different musical structures. The pairwise similarity engine (Component B) is genre-agnostic in principle, but its calibration has not been validated outside the European classical tradition. The Component A/B synthesis in Section 5.6 deliberately applies a classical-trained style-fingerprint engine to non-classical recordings, precisely to test whether a general, genre-blind notion of compositional style says anything useful about these disputes. That it produces internally consistent, interpretable results (Table 5.6) is encouraging, but the same genre-mismatch caveat that applies to Component A's classifier applies here too.

---

## 7. Conclusion

This paper presents a two-component framework for analyzing musical style and copyright similarity. Component A shows that composer identity can be reliably inferred from 21 numerical features extracted from MIDI files, reaching 88.7% test accuracy (Random Forest; an SVM reached 91.5%) on a six-composer task. The most predictive features, pitch range, key stability, and average pitch, align with established music-theoretical understanding of each composer's style. One unexpected generalization result, an AI-generated Carnatic piece classified as Bach, reveals a cultural scope limitation that matters for deploying such tools on diverse music.

Component B introduces a pairwise similarity engine measuring melodic, harmonic, rhythmic, and n-gram similarity between works in real copyright disputes, applied across 43 cases (33 fully scored) spanning 1946 to 2023. A logistic regression model finds that non-musical case factors, meaning plaintiff fame, defendant commercial success, expert testimony, and litigation forum, predict case outcomes far more reliably than musical similarity alone (AUC 0.86 versus 0.39 under cross-validation), with plaintiff fame the single strongest predictor. One methodological choice worth noting is the switch from absolute-pitch comparison to interval-based comparison, which makes the melodic metrics key-independent and lets them detect shared melodic content that a naive pitch comparison would miss across a transposed key.

Five outlier cases, where computed similarity most diverges from the actual ruling, were analyzed in depth and cross-checked against an independent measure: Component A's style-fingerprinting engine, built to distinguish classical composers and applied here to disputed pop, rock, and musical-theater recordings it was never designed to evaluate. The clearest result is *Williams v. Gaye* ("Blurred Lines"), the case with the highest-profile infringement verdict in the dataset and, at the same time, the case where both Component B's specialized engine and Component A's general engine find the least objective similarity of any pair examined. Two independently built tools, agreeing with neither each other's design goals nor the case's outcome, both find less measurable similarity than the verdict implied. That is a partial corroboration of the widespread criticism that the verdict extended copyright to an unprotectable "feel" rather than to specific borrowed expression.

The central finding is that computational similarity and legal liability measure related but distinct things, and that, at least across the cases collected here, the gap between them is better explained by who is litigating than by what the music sounds like. Legal doctrine applies a protectability and access filter on top of similarity that no acoustic measurement can replicate, and non-musical facts about the parties explain more of the variance in outcomes than any musical measurement tested. Mapping exactly where and why individual cases depart from that pattern, as the five case studies begin to do, is the direction I intend to take this project as the dataset and methods grow.

---

## References

Arnstein v. Porter, 154 F.2d 464 (2d Cir. 1946).

Backer, E., & van Kranenburg, P. (2005). On Musical Stylometry: A Pattern Recognition Approach. *Pattern Recognition Letters*, 26(3), 299–309.

Bittner, R. M., Bosch, J. J., Rubinstein, D., Meseguer-Brocal, G., & Ewert, S. (2022). A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation. *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

Bridgeport Music, Inc. v. Dimension Films, 410 F.3d 792 (6th Cir. 2005).

Bright Tunes Music Corp. v. Harrisongs Music, Ltd., 420 F. Supp. 177 (S.D.N.Y. 1976).

Cronin, C. (1998). Concepts of Melodic Similarity in Music-Copyright Infringement Suits. In W. B. Hewlett & E. Selfridge-Field (Eds.), *Computing in Musicology 11* (pp. 187–209). MIT Press.

Cuthbert, M. S., & Ariza, C. (2010). music21: A Toolkit for Computer-Aided Musicology and Symbolic Music Data. *ISMIR 2010*.

Fishman, J. A. (2018). Music as a Matter of Law. *Harvard Law Review*, 131(6), 1861–1921.

Gray v. Perry, 2020 WL 1275221 (C.D. Cal. 2020).

Honing, H. (2013). Structure and Interpretation of Rhythm in Music. In D. Deutsch (Ed.), *The Psychology of Music* (3rd ed., pp. 369–404). Academic Press.

Müller, M. (2015). *Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications*. Springer.

Raffel, C. (2016). *Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching* (Doctoral dissertation). Columbia University.

Repp v. Webber, 858 F. Supp. 1292 (S.D.N.Y. 1994), *rev'd*, 132 F.3d 882 (2d Cir. 1997).

Sheeran v. Chokri, [2022] EWHC 827 (Ch).

Skidmore v. Led Zeppelin, 952 F.3d 1051 (9th Cir. 2020).

Swirsky v. Carey, 376 F.3d 841 (9th Cir. 2004).

VMG Salsoul, LLC v. Ciccone, 824 F.3d 871 (9th Cir. 2016).

Williams v. Gaye, 895 F.3d 1106 (9th Cir. 2018).

Yuan, Y., Cronin, C., Müllensiefen, D., Fujii, S., & Savage, P. E. (2023). Perceptual and Automated Estimates of Infringement in 40 Music Copyright Cases. *Transactions of the International Society for Music Information Retrieval*, 6(1), 117–134.
