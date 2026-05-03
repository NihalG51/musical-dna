"""
Musical DNA — Feature Extraction Pipeline
==========================================
Extracts 25 musical features from MIDI files for composer classification
and copyright similarity analysis.

Features are grouped into 5 categories:
  1. Melodic (6 features)
  2. Harmonic (5 features)
  3. Rhythmic (5 features)
  4. Structural (5 features)
  5. Pairwise Similarity (4 features) — used in Component B

Author: Nihal
"""

import numpy as np
from collections import Counter
from music21 import converter, analysis, pitch, interval, key


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_midi(filepath):
    """Load a MIDI file and return a music21 Score object.
    
    Args:
        filepath: Path to MIDI file
        
    Returns:
        music21.stream.Score object, or None if file can't be parsed
    """
    try:
        score = converter.parse(filepath)
        return score
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_notes(score):
    """Extract all Note objects from a score (flattened, no rests).
    
    Args:
        score: music21.stream.Score
        
    Returns:
        List of music21.note.Note objects
    """
    return [n for n in score.flatten().notes if n.isNote]


def get_all_notes_and_chords(score):
    """Extract all Notes and Chords from a score (flattened).
    
    Returns individual pitches from chords as well.
    """
    flat = score.flatten().notes
    pitches = []
    for element in flat:
        if element.isNote:
            pitches.append(element)
        elif element.isChord:
            for p in element.pitches:
                pitches.append(p)
    return pitches


# =============================================================================
# CATEGORY 1: MELODIC FEATURES (Week 1: features 1-3, Week 2: features 4-6)
# =============================================================================

def pitch_range(score):
    """Feature 1: Pitch range — highest minus lowest MIDI note number.
    
    A wide pitch range often indicates virtuosic or dramatic writing.
    Bach fugues tend to have moderate range; Rachmaninoff spans the full keyboard.
    
    Returns:
        int: Range in semitones (0-88 for piano)
    """
    notes = get_notes(score)
    if not notes:
        return 0
    midi_values = [n.pitch.midi for n in notes]
    return max(midi_values) - min(midi_values)


def average_pitch(score):
    """Feature 2: Average pitch — mean MIDI note value.
    
    Higher values = piece sits higher on the keyboard.
    Chopin tends toward mid-high register; Bach organ works sit lower.
    
    Returns:
        float: Mean MIDI note value (roughly 21-108 for piano)
    """
    notes = get_notes(score)
    if not notes:
        return 0.0
    return float(np.mean([n.pitch.midi for n in notes]))


def pitch_class_entropy(score):
    """Feature 3: Pitch class entropy — how evenly all 12 pitch classes are used.
    
    Shannon entropy of the pitch class distribution (0-12 scale mapped to 0.0-1.0).
    High entropy = uses all 12 notes roughly equally (chromatic, like Bach fugues).
    Low entropy = sticks to a few notes (simpler tonal writing).
    
    Returns:
        float: Normalized Shannon entropy (0.0 to 1.0)
    """
    notes = get_notes(score)
    if not notes:
        return 0.0
    
    # Count occurrences of each pitch class (0-11: C, C#, D, ..., B)
    pitch_classes = [n.pitch.pitchClass for n in notes]
    counts = Counter(pitch_classes)
    total = sum(counts.values())
    
    # Shannon entropy: H = -sum(p * log2(p))
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    # Normalize: max entropy for 12 classes = log2(12) ≈ 3.585
    max_entropy = np.log2(12)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def interval_histogram(score):
    """Feature 4: Interval histogram — distribution of melodic intervals.
    
    Counts the frequency of each interval size (in semitones) between
    consecutive notes. Returns the histogram as a dict.
    
    Returns:
        dict: {interval_in_semitones: count}
    """
    notes = get_notes(score)
    if len(notes) < 2:
        return {}
    
    intervals = []
    for i in range(1, len(notes)):
        semitones = abs(notes[i].pitch.midi - notes[i-1].pitch.midi)
        intervals.append(semitones)
    
    return dict(Counter(intervals))


def leap_ratio(score):
    """Feature 5: Leap ratio — percentage of intervals larger than a major 2nd.
    
    A "leap" is any interval > 2 semitones (i.e., larger than a whole step).
    Conjunct (stepwise) motion = low leap ratio. Disjunct = high leap ratio.
    Chopin tends toward ornamental stepwise motion; Beethoven uses dramatic leaps.
    
    Returns:
        float: Proportion of leaps (0.0 to 1.0)
    """
    notes = get_notes(score)
    if len(notes) < 2:
        return 0.0
    
    leaps = 0
    total = 0
    for i in range(1, len(notes)):
        semitones = abs(notes[i].pitch.midi - notes[i-1].pitch.midi)
        total += 1
        if semitones > 2:
            leaps += 1
    
    return leaps / total if total > 0 else 0.0


def melodic_contour(score):
    """Feature 6: Melodic contour — ratio of ascending to descending intervals.
    
    Values > 1.0 mean more ascending motion; < 1.0 means more descending.
    Returns the ratio as ascending / (ascending + descending) for 0-1 range.
    
    Returns:
        float: Proportion of ascending intervals (0.0 to 1.0)
    """
    notes = get_notes(score)
    if len(notes) < 2:
        return 0.5
    
    ascending = 0
    descending = 0
    for i in range(1, len(notes)):
        diff = notes[i].pitch.midi - notes[i-1].pitch.midi
        if diff > 0:
            ascending += 1
        elif diff < 0:
            descending += 1
    
    total = ascending + descending
    return ascending / total if total > 0 else 0.5


# =============================================================================
# CATEGORY 2: HARMONIC FEATURES (Week 2)
# =============================================================================

def key_stability(score):
    """Feature 7: Key stability — how firmly a piece stays in its key.
    
    Uses the Krumhansl-Schmuckler key-finding algorithm's correlation score.
    High correlation = very clearly in one key. Low = ambiguous tonality.
    Debussy will score lower than Mozart.
    
    Returns:
        float: Key correlation coefficient (typically 0.3 to 0.95)
    """
    try:
        key_result = score.analyze('key')
        return float(key_result.correlationCoefficient)
    except Exception:
        return 0.0


def modulation_frequency(score):
    """Feature 8: Modulation frequency — how often the key changes.
    
    Analyzes key in a sliding window across the piece and counts transitions.
    Normalized by piece length.
    
    Returns:
        float: Key changes per 10 measures
    """
    try:
        measures = score.parts[0].getElementsByClass('Measure')
        if len(measures) < 4:
            return 0.0
        
        window_size = 4  # analyze in 4-measure windows
        keys_found = []
        
        for i in range(0, len(measures) - window_size + 1, 2):
            excerpt = score.measures(i + 1, i + window_size)
            try:
                k = excerpt.analyze('key')
                keys_found.append(str(k))
            except Exception:
                continue
        
        # Count key changes
        changes = 0
        for i in range(1, len(keys_found)):
            if keys_found[i] != keys_found[i-1]:
                changes += 1
        
        # Normalize per 10 measures
        num_measures = len(measures)
        return (changes / num_measures) * 10 if num_measures > 0 else 0.0
    except Exception:
        return 0.0


def chord_vocabulary_size(score):
    """Feature 9: Chord vocabulary size — number of unique chord types used.
    
    Chordifies the score and counts distinct chord types (e.g., major triad,
    minor seventh, diminished, etc.).
    
    Returns:
        int: Count of unique chord types
    """
    try:
        chords = score.chordify()
        chord_types = set()
        for c in chords.flatten().getElementsByClass('Chord'):
            chord_types.add(c.commonName)
        return len(chord_types)
    except Exception:
        return 0


def dissonance_ratio(score):
    """Feature 10: Dissonance ratio — proportion of dissonant intervals.
    
    Counts minor 2nds, major 7ths, and tritones as dissonant intervals
    among all vertical (simultaneous) intervals.
    
    Returns:
        float: Proportion of dissonant intervals (0.0 to 1.0)
    """
    try:
        chords = score.chordify()
        dissonant = 0
        total = 0
        
        dissonant_semitones = {1, 2, 6, 10, 11}  # m2, M2, tritone, m7, M7
        
        for c in chords.flatten().getElementsByClass('Chord'):
            pitches = c.pitches
            for i in range(len(pitches)):
                for j in range(i + 1, len(pitches)):
                    semitones = abs(pitches[j].midi - pitches[i].midi) % 12
                    total += 1
                    if semitones in dissonant_semitones:
                        dissonant += 1
        
        return dissonant / total if total > 0 else 0.0
    except Exception:
        return 0.0


def tonal_gravity(score):
    """Feature 11: Tonal gravity — tendency to resolve to tonic.
    
    Measures the frequency of V→I and viio→I chord progressions,
    which represent strong cadential motion. Higher = more tonal pull.
    
    Returns:
        float: Cadential progressions per 10 chords
    """
    try:
        k = score.analyze('key')
        chords_stream = score.chordify()
        chord_list = list(chords_stream.flatten().getElementsByClass('Chord'))
        
        if len(chord_list) < 2:
            return 0.0
        
        cadential_count = 0
        for i in range(1, len(chord_list)):
            try:
                rn_prev = analysis.roman.romanNumeralFromChord(chord_list[i-1], k)
                rn_curr = analysis.roman.romanNumeralFromChord(chord_list[i], k)
                
                prev_fig = rn_prev.romanNumeralAlone
                curr_fig = rn_curr.romanNumeralAlone
                
                # V→I or viio→I progressions
                if curr_fig == 'I' and prev_fig in ('V', 'viio', 'vii'):
                    cadential_count += 1
            except Exception:
                continue
        
        return (cadential_count / len(chord_list)) * 10 if chord_list else 0.0
    except Exception:
        return 0.0


# =============================================================================
# CATEGORY 3: RHYTHMIC FEATURES (Week 2)
# =============================================================================

def note_density(score):
    """Feature 12: Note density — notes per beat.
    
    Higher density = busier, more virtuosic texture.
    Chopin etudes will score very high; simple chorale movements lower.
    
    Returns:
        float: Average notes per quarter-note beat
    """
    notes = get_notes(score)
    if not notes:
        return 0.0
    
    total_duration = float(score.flatten().highestTime)
    return len(notes) / total_duration if total_duration > 0 else 0.0


def duration_variance(score):
    """Feature 13: Duration variance — how varied the note lengths are.
    
    Low variance = steady rhythms (marches, Bach inventions).
    High variance = mixed rhythms (Chopin rubato, Debussy).
    
    Returns:
        float: Variance of note durations in quarter-note lengths
    """
    notes = get_notes(score)
    if not notes:
        return 0.0
    
    durations = [float(n.duration.quarterLength) for n in notes]
    return float(np.var(durations))


def syncopation_index(score):
    """Feature 14: Syncopation index — proportion of notes on weak beats.
    
    Notes starting on offbeats (not on beat 1 or 3 in 4/4) count as syncopated.
    
    Returns:
        float: Proportion of syncopated notes (0.0 to 1.0)
    """
    notes = get_notes(score)
    if not notes:
        return 0.0
    
    syncopated = 0
    for n in notes:
        beat = float(n.beat)
        # In most time signatures, beats 1 and 3 are strong
        if beat % 1.0 != 0 or beat % 2.0 != 1.0:
            syncopated += 1
    
    return syncopated / len(notes)


def rest_ratio(score):
    """Feature 15: Rest ratio — proportion of the piece that is silence.
    
    Returns:
        float: Total rest duration / total piece duration (0.0 to 1.0)
    """
    flat = score.flatten()
    rests = flat.getElementsByClass('Rest')
    
    total_duration = float(flat.highestTime)
    if total_duration == 0:
        return 0.0
    
    rest_duration = sum(float(r.duration.quarterLength) for r in rests)
    return rest_duration / total_duration


def rhythmic_entropy(score):
    """Feature 16: Rhythmic entropy — predictability of rhythm patterns.
    
    Shannon entropy of the note-duration histogram.
    High entropy = many different duration values used.
    Low entropy = rhythmically uniform.
    
    Returns:
        float: Normalized Shannon entropy (0.0 to 1.0)
    """
    notes = get_notes(score)
    if not notes:
        return 0.0
    
    # Quantize durations to nearest sixteenth note to reduce noise
    durations = [round(float(n.duration.quarterLength) * 4) / 4 for n in notes]
    counts = Counter(durations)
    total = sum(counts.values())
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    # Normalize by max possible entropy
    num_unique = len(counts)
    max_entropy = np.log2(num_unique) if num_unique > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


# =============================================================================
# CATEGORY 4: STRUCTURAL FEATURES (Week 2)
# =============================================================================

def piece_length(score):
    """Feature 17: Piece length — total number of measures.
    
    Returns:
        int: Number of measures in the first part
    """
    try:
        measures = score.parts[0].getElementsByClass('Measure')
        return len(measures)
    except Exception:
        return 0


def repetition_ratio(score):
    """Feature 18: Repetition ratio — self-similarity of 4-bar chunks.
    
    Computes feature vectors for each 4-bar window and measures
    average cosine similarity between consecutive windows.
    
    Returns:
        float: Mean cosine similarity (0.0 to 1.0)
    """
    try:
        measures = list(score.parts[0].getElementsByClass('Measure'))
        if len(measures) < 8:
            return 0.0
        
        from scipy.spatial.distance import cosine
        
        window_features = []
        for i in range(0, len(measures) - 3, 4):
            window = score.measures(i + 1, i + 4)
            notes = get_notes(window)
            if not notes:
                continue
            
            # Simple feature vector: mean pitch, note count, mean duration
            feat = [
                np.mean([n.pitch.midi for n in notes]),
                len(notes),
                np.mean([float(n.duration.quarterLength) for n in notes])
            ]
            window_features.append(feat)
        
        if len(window_features) < 2:
            return 0.0
        
        similarities = []
        for i in range(1, len(window_features)):
            try:
                sim = 1 - cosine(window_features[i-1], window_features[i])
                similarities.append(sim)
            except Exception:
                continue
        
        return float(np.mean(similarities)) if similarities else 0.0
    except Exception:
        return 0.0


def dynamic_range(score):
    """Feature 19: Dynamic range — variation in velocity values.
    
    MIDI velocity ranges 0-127. Wider range = more dynamic contrast.
    Returns 0 if velocity data is not available.
    
    Returns:
        int: max(velocity) - min(velocity)
    """
    notes = get_notes(score)
    if not notes:
        return 0
    
    velocities = []
    for n in notes:
        vol = n.volume
        if vol and vol.velocity is not None:
            velocities.append(vol.velocity)
    
    if not velocities:
        return 0
    return max(velocities) - min(velocities)


def voice_count(score):
    """Feature 20: Voice count — number of simultaneous parts.
    
    Returns:
        int: Number of parts in the score
    """
    try:
        return len(score.parts)
    except Exception:
        return 1


def texture_density(score):
    """Feature 21: Texture density — average simultaneous notes per beat.
    
    Measures how "thick" the texture is on average.
    
    Returns:
        float: Mean notes sounding at each beat position
    """
    try:
        chords = score.chordify()
        chord_list = list(chords.flatten().getElementsByClass('Chord'))
        
        if not chord_list:
            return 1.0
        
        sizes = [len(c.pitches) for c in chord_list]
        return float(np.mean(sizes))
    except Exception:
        return 1.0


# =============================================================================
# MAIN PIPELINE
# =============================================================================

# Feature names in order (for DataFrame columns)
FEATURE_NAMES = [
    # Melodic (6)
    'pitch_range', 'average_pitch', 'pitch_class_entropy',
    'leap_ratio', 'melodic_contour', 'interval_histogram_unique_count',
    # Harmonic (5)
    'key_stability', 'modulation_frequency', 'chord_vocabulary_size',
    'dissonance_ratio', 'tonal_gravity',
    # Rhythmic (5)
    'note_density', 'duration_variance', 'syncopation_index',
    'rest_ratio', 'rhythmic_entropy',
    # Structural (5)
    'piece_length', 'repetition_ratio', 'dynamic_range',
    'voice_count', 'texture_density',
]


def extract_features(score):
    """Extract all 21 solo-piece features from a music21 Score.
    
    Args:
        score: music21.stream.Score object
        
    Returns:
        dict: {feature_name: value} for all features
    """
    features = {}
    
    # Melodic
    features['pitch_range'] = safe_extract(pitch_range, score)
    features['average_pitch'] = safe_extract(average_pitch, score)
    features['pitch_class_entropy'] = safe_extract(pitch_class_entropy, score)
    features['leap_ratio'] = safe_extract(leap_ratio, score)
    features['melodic_contour'] = safe_extract(melodic_contour, score)
    
    # For interval histogram, we store the count of unique intervals
    hist = safe_extract(interval_histogram, score)
    features['interval_histogram_unique_count'] = len(hist) if isinstance(hist, dict) else 0
    
    # Harmonic
    features['key_stability'] = safe_extract(key_stability, score)
    features['modulation_frequency'] = safe_extract(modulation_frequency, score)
    features['chord_vocabulary_size'] = safe_extract(chord_vocabulary_size, score)
    features['dissonance_ratio'] = safe_extract(dissonance_ratio, score)
    features['tonal_gravity'] = safe_extract(tonal_gravity, score)
    
    # Rhythmic
    features['note_density'] = safe_extract(note_density, score)
    features['duration_variance'] = safe_extract(duration_variance, score)
    features['syncopation_index'] = safe_extract(syncopation_index, score)
    features['rest_ratio'] = safe_extract(rest_ratio, score)
    features['rhythmic_entropy'] = safe_extract(rhythmic_entropy, score)
    
    # Structural
    features['piece_length'] = safe_extract(piece_length, score)
    features['repetition_ratio'] = safe_extract(repetition_ratio, score)
    features['dynamic_range'] = safe_extract(dynamic_range, score)
    features['voice_count'] = safe_extract(voice_count, score)
    features['texture_density'] = safe_extract(texture_density, score)
    
    return features


def safe_extract(func, score):
    """Safely extract a feature, returning 0 on any error.
    
    This is crucial — some MIDI files have quirks that crash specific features.
    We'd rather get 0 for one feature than lose the entire file.
    """
    try:
        result = func(score)
        return result
    except Exception as e:
        print(f"  Warning: {func.__name__} failed: {e}")
        return 0


def extract_features_from_file(filepath):
    """Load a MIDI file and extract all features.
    
    Args:
        filepath: Path to MIDI file
        
    Returns:
        dict or None: Feature dictionary, or None if file can't be loaded
    """
    score = load_midi(filepath)
    if score is None:
        return None
    return extract_features(score)
