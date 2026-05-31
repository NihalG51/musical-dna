"""
Musical DNA — Pairwise Similarity Features
============================================
Computes four pairwise similarity metrics between two MIDI files.
Used for Component B (copyright case scoring).

Features:
  1. melodic_similarity      — normalized sequence match on pitch sequences
  2. harmonic_cosine_sim     — cosine similarity of chord-type frequency vectors
  3. rhythmic_dtw_sim        — DTW similarity on inter-onset interval sequences
  4. ngram_overlap           — Jaccard index of 4-note melodic n-grams

All scores are in [0, 1] where 1 = identical and 0 = completely dissimilar.
"""

import numpy as np
from collections import Counter
from difflib import SequenceMatcher
from scipy.spatial.distance import cosine

from .features import load_midi, get_notes


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _get_pitch_sequence(score):
    notes = get_notes(score)
    return [n.pitch.midi for n in notes]


def _get_ioi_sequence(score, max_len=500):
    """Inter-onset intervals (in quarter-note beats), truncated for DTW speed."""
    notes = get_notes(score)
    if not notes:
        return np.array([])
    onsets = np.array([float(n.offset) for n in notes[:max_len]])
    return np.diff(onsets) if len(onsets) > 1 else np.array([0.0])


def _dtw_distance(seq1, seq2):
    """O(n·m) dynamic time warping distance."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _get_chord_counter(score):
    try:
        chords = score.chordify()
        return Counter(
            c.commonName
            for c in chords.flatten().getElementsByClass('Chord')
        )
    except Exception:
        return Counter()


def _safe_pairwise(func, *args):
    try:
        result = func(*args)
        return float(np.clip(result, 0.0, 1.0))
    except Exception:
        return 0.0


# =============================================================================
# PAIRWISE FEATURES
# =============================================================================

def melodic_similarity(score1, score2):
    """Feature 1: Normalized sequence match on MIDI pitch sequences.

    Uses SequenceMatcher ratio = 2*M/T (M = matching elements, T = total).
    Captures shared melodic runs and passages, not just interval patterns.

    Returns:
        float: similarity in [0, 1]
    """
    p1 = _get_pitch_sequence(score1)
    p2 = _get_pitch_sequence(score2)
    if not p1 or not p2:
        return 0.0
    return SequenceMatcher(None, p1, p2).ratio()


def harmonic_cosine_sim(score1, score2):
    """Feature 2: Cosine similarity of chord-type frequency vectors.

    Chordifies each score, builds a frequency histogram of chord types
    (e.g. 'major triad', 'minor seventh'), then computes cosine similarity.
    Captures shared harmonic vocabulary regardless of key.

    Returns:
        float: similarity in [0, 1]
    """
    c1 = _get_chord_counter(score1)
    c2 = _get_chord_counter(score2)
    all_types = sorted(set(c1) | set(c2))
    if not all_types:
        return 0.0
    v1 = np.array([c1.get(t, 0) for t in all_types], dtype=float)
    v2 = np.array([c2.get(t, 0) for t in all_types], dtype=float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(1 - cosine(v1, v2))


def rhythmic_dtw_sim(score1, score2, max_len=300):
    """Feature 3: DTW similarity on inter-onset interval sequences.

    Aligns the rhythmic pattern of both pieces using dynamic time warping
    on inter-onset intervals (IOIs). Lower DTW distance = higher similarity.
    Sequences truncated to max_len for speed.

    Returns:
        float: similarity in [0, 1]
    """
    ioi1 = _get_ioi_sequence(score1, max_len)
    ioi2 = _get_ioi_sequence(score2, max_len)
    if len(ioi1) == 0 or len(ioi2) == 0:
        return 0.0
    dist = _dtw_distance(ioi1, ioi2)
    # Normalize by average sequence length × max possible IOI difference
    avg_len = (len(ioi1) + len(ioi2)) / 2
    max_ioi = max(ioi1.max(), ioi2.max()) if avg_len > 0 else 1.0
    normalized = dist / (avg_len * max_ioi) if avg_len * max_ioi > 0 else 0.0
    return max(0.0, 1.0 - normalized)


def ngram_overlap(score1, score2, n=4):
    """Feature 4: Jaccard index of n-note melodic n-grams.

    Extracts all n-length subsequences of MIDI pitches from each piece
    and computes intersection / union (Jaccard index).
    Captures shared melodic phrases independent of order or position.

    Returns:
        float: Jaccard similarity in [0, 1]
    """
    p1 = _get_pitch_sequence(score1)
    p2 = _get_pitch_sequence(score2)
    if len(p1) < n or len(p2) < n:
        return 0.0

    def make_ngrams(seq):
        return set(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

    ng1 = make_ngrams(p1)
    ng2 = make_ngrams(p2)
    union = len(ng1 | ng2)
    return len(ng1 & ng2) / union if union > 0 else 0.0


# =============================================================================
# MASTER FUNCTION
# =============================================================================

def compute_pairwise_similarity(filepath1, filepath2):
    """Load two MIDI files and compute all four pairwise similarity scores.

    Args:
        filepath1: path to plaintiff's MIDI file
        filepath2: path to defendant's MIDI file

    Returns:
        dict with keys: melodic_similarity, harmonic_cosine_sim,
                        rhythmic_dtw_sim, ngram_overlap_4
        or None if either file fails to load.
    """
    score1 = load_midi(filepath1)
    score2 = load_midi(filepath2)
    if score1 is None or score2 is None:
        return None

    return {
        'melodic_similarity':   _safe_pairwise(melodic_similarity, score1, score2),
        'harmonic_cosine_sim':  _safe_pairwise(harmonic_cosine_sim, score1, score2),
        'rhythmic_dtw_sim':     _safe_pairwise(rhythmic_dtw_sim, score1, score2),
        'ngram_overlap_4':      _safe_pairwise(ngram_overlap, score1, score2, 4),
    }
