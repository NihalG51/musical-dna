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

def _get_melody_notes(score, top_pct=0.30):
    """Return the top-pitched notes as a melody proxy.

    Basic Pitch outputs single-track MIDI from full-band audio, so
    part-based separation doesn't apply. Instead, take the top `top_pct`
    fraction of notes by pitch — in a full-band mix the melody typically
    sits above the accompaniment in pitch space.

    For multi-part scores (e.g. Piano Solo MIDIs), picks the highest-pitch
    part first, then applies the top-pct filter within that part.
    """
    parts = score.parts
    if len(parts) > 1:
        def avg_pitch(part):
            notes = [n for n in part.flatten().notes if n.isNote]
            return sum(n.pitch.midi for n in notes) / len(notes) if notes else 0
        source = max(parts, key=avg_pitch)
        all_notes = [n for n in source.flatten().notes if n.isNote]
    else:
        all_notes = get_notes(score)

    if not all_notes:
        return []

    # Keep only the top top_pct by pitch, sorted by onset time
    cutoff = int(len(all_notes) * (1 - top_pct))
    by_pitch = sorted(all_notes, key=lambda n: n.pitch.midi)
    top_notes = set(id(n) for n in by_pitch[cutoff:])
    return sorted([n for n in all_notes if id(n) in top_notes],
                  key=lambda n: n.offset)


MELODY_SAMPLE = 200   # max notes used for melodic/n-gram comparison


def _get_pitch_sequence(score):
    notes = _get_melody_notes(score)
    # Evenly sample to MELODY_SAMPLE notes so dense files don't dominate
    if len(notes) > MELODY_SAMPLE:
        step = len(notes) / MELODY_SAMPLE
        notes = [notes[int(i * step)] for i in range(MELODY_SAMPLE)]
    return [n.pitch.midi for n in notes]


def _get_interval_sequence(score):
    """Convert pitch sequence to melodic interval sequence (semitone differences).

    Key-independent: the same melody in any transposition produces identical
    intervals. E.g. 'He's So Fine' in F and 'My Sweet Lord' in E both yield
    the same [+2, +2, +3, ...] pattern for their shared hook.
    """
    pitches = _get_pitch_sequence(score)
    if len(pitches) < 2:
        return []
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def _get_ioi_sequence(score, max_len=500):
    """Inter-onset intervals (in quarter-note beats), truncated for DTW speed."""
    notes = _get_melody_notes(score)
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
    """Feature 1: Normalized sequence match on melodic interval sequences.

    Compares semitone-difference sequences rather than absolute pitches so
    the same melody in different keys scores as similar (key-independent).
    Uses SequenceMatcher ratio = 2*M/T.

    Returns:
        float: similarity in [0, 1]
    """
    i1 = _get_interval_sequence(score1)
    i2 = _get_interval_sequence(score2)
    if not i1 or not i2:
        return 0.0
    return SequenceMatcher(None, i1, i2).ratio()


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
    # Normalize by avg_len × mean_ioi (typical step cost).
    # mean_ioi is a much tighter reference than max_ioi, spreading scores
    # across the full [0,1] range instead of compressing them to ~0.95.
    avg_len  = (len(ioi1) + len(ioi2)) / 2
    mean_ioi = (ioi1.mean() + ioi2.mean()) / 2 if avg_len > 0 else 1.0
    normalizer = avg_len * mean_ioi if avg_len * mean_ioi > 0 else 1.0
    return max(0.0, 1.0 - dist / normalizer)


def ngram_overlap(score1, score2, n=4):
    """Feature 4: Jaccard index of n-interval melodic n-grams.

    Extracts all n-length subsequences of melodic *intervals* (semitone
    differences) and computes intersection / union (Jaccard index).
    Key-independent: the same melodic phrase in any transposition matches.

    Returns:
        float: Jaccard similarity in [0, 1]
    """
    i1 = _get_interval_sequence(score1)
    i2 = _get_interval_sequence(score2)
    if len(i1) < n or len(i2) < n:
        return 0.0

    def make_ngrams(seq):
        return set(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

    ng1 = make_ngrams(i1)
    ng2 = make_ngrams(i2)
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
