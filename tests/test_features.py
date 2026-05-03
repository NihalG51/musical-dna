"""
Musical DNA — Feature Extraction Test
======================================
Tests the first 5 features using music21's built-in corpus.
No external MIDI files needed — this uses Bach and Mozart pieces
that ship with the music21 library.

Run this on Day 1 to verify your environment is set up correctly:
    python tests/test_features.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from music21 import corpus
from src.features import (
    load_midi, get_notes,
    pitch_range, average_pitch, pitch_class_entropy,
    leap_ratio, melodic_contour,
    key_stability, note_density,
    extract_features, FEATURE_NAMES
)


def test_environment():
    """Test that all required packages are installed."""
    print("Testing environment setup...")
    
    packages = {
        'music21': 'music21',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    all_ok = True
    for module, pip_name in packages.items():
        try:
            __import__(module)
            print(f"  OK  {pip_name}")
        except ImportError:
            print(f"  FAIL  {pip_name} — run: pip install {pip_name}")
            all_ok = False
    
    return all_ok


def test_corpus_loading():
    """Test that we can load pieces from music21's built-in corpus."""
    print("\nTesting corpus loading...")
    
    # Load a Bach piece from the built-in corpus
    print("  Loading Bach BWV 66.6 (chorale)...")
    bach = corpus.parse('bach/bwv66.6')
    print(f"  OK  Loaded: {bach.metadata.title or 'BWV 66.6'}")
    print(f"       Parts: {len(bach.parts)}")
    notes = get_notes(bach)
    print(f"       Notes: {len(notes)}")
    
    # Load a Mozart piece
    print("  Loading Mozart K545 (Piano Sonata)...")
    mozart = corpus.parse('mozart/k545/movement1_exposition.mxl')
    print(f"  OK  Loaded: Mozart K545 Movement 1")
    print(f"       Parts: {len(mozart.parts)}")
    notes_m = get_notes(mozart)
    print(f"       Notes: {len(notes_m)}")
    
    return bach, mozart


def test_week1_features(bach, mozart):
    """Test the 5 Week 1 features on Bach and Mozart pieces."""
    print("\nTesting Week 1 features (first 5)...")
    print("-" * 60)
    
    # Feature 1: Pitch Range
    bach_pr = pitch_range(bach)
    mozart_pr = pitch_range(mozart)
    print(f"\n  1. Pitch Range")
    print(f"     Bach BWV 66.6:   {bach_pr} semitones")
    print(f"     Mozart K545:     {mozart_pr} semitones")
    print(f"     Makes sense? {'Yes' if bach_pr > 0 and mozart_pr > 0 else 'CHECK THIS'}")
    
    # Feature 2: Average Pitch
    bach_ap = average_pitch(bach)
    mozart_ap = average_pitch(mozart)
    print(f"\n  2. Average Pitch (MIDI note number)")
    print(f"     Bach BWV 66.6:   {bach_ap:.1f} (≈{midi_to_note(bach_ap)})")
    print(f"     Mozart K545:     {mozart_ap:.1f} (≈{midi_to_note(mozart_ap)})")
    
    # Feature 3: Pitch Class Entropy
    bach_pce = pitch_class_entropy(bach)
    mozart_pce = pitch_class_entropy(mozart)
    print(f"\n  3. Pitch Class Entropy (0-1, higher = more chromatic)")
    print(f"     Bach BWV 66.6:   {bach_pce:.3f}")
    print(f"     Mozart K545:     {mozart_pce:.3f}")
    print(f"     Bach more chromatic? {'Yes' if bach_pce > mozart_pce else 'Interesting — check the pieces'}")
    
    # Feature 4: Leap Ratio
    bach_lr = leap_ratio(bach)
    mozart_lr = leap_ratio(mozart)
    print(f"\n  4. Leap Ratio (0-1, higher = more jumps)")
    print(f"     Bach BWV 66.6:   {bach_lr:.3f}")
    print(f"     Mozart K545:     {mozart_lr:.3f}")
    
    # Feature 5: Melodic Contour
    bach_mc = melodic_contour(bach)
    mozart_mc = melodic_contour(mozart)
    print(f"\n  5. Melodic Contour (0-1, >0.5 = mostly ascending)")
    print(f"     Bach BWV 66.6:   {bach_mc:.3f}")
    print(f"     Mozart K545:     {mozart_mc:.3f}")
    
    print("\n" + "-" * 60)
    print("All 5 Week 1 features extracted successfully!")
    
    return True


def test_full_pipeline(bach):
    """Test the full 21-feature extraction pipeline."""
    print("\nTesting full pipeline (all 21 features)...")
    
    features = extract_features(bach)
    
    print(f"  Extracted {len(features)} features from Bach BWV 66.6:")
    for name, value in features.items():
        if isinstance(value, float):
            print(f"    {name:35s} = {value:.4f}")
        else:
            print(f"    {name:35s} = {value}")
    
    # Verify we got all expected features
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        print(f"\n  WARNING: Missing features: {missing}")
        return False
    
    print(f"\n  All {len(FEATURE_NAMES)} features present. Pipeline works!")
    return True


def test_dataframe_output(bach, mozart):
    """Test that features can be organized into a pandas DataFrame."""
    import pandas as pd
    
    print("\nTesting DataFrame output...")
    
    bach_features = extract_features(bach)
    bach_features['composer'] = 'bach'
    bach_features['filename'] = 'bwv66.6'
    
    mozart_features = extract_features(mozart)
    mozart_features['composer'] = 'mozart'
    mozart_features['filename'] = 'k545_mvt1'
    
    df = pd.DataFrame([bach_features, mozart_features])
    
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Preview:")
    print(df[['composer', 'pitch_range', 'average_pitch', 'pitch_class_entropy', 
              'key_stability', 'note_density']].to_string(index=False))
    
    return True


def midi_to_note(midi_num):
    """Convert MIDI note number to approximate note name."""
    from music21 import pitch as m21pitch
    try:
        p = m21pitch.Pitch(midi=int(round(midi_num)))
        return str(p)
    except Exception:
        return f"MIDI {midi_num:.0f}"


def main():
    print("=" * 60)
    print("  MUSICAL DNA — Feature Extraction Test Suite")
    print("=" * 60)
    
    # Step 1: Check environment
    if not test_environment():
        print("\nFix the missing packages above and re-run.")
        sys.exit(1)
    
    # Step 2: Load test pieces
    bach, mozart = test_corpus_loading()
    
    # Step 3: Test Week 1 features
    test_week1_features(bach, mozart)
    
    # Step 4: Test full pipeline
    test_full_pipeline(bach)
    
    # Step 5: Test DataFrame output
    test_dataframe_output(bach, mozart)
    
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED — Your environment is ready!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Download MIDI files into data/midi/<composer>/ folders")
    print("  2. Run: python src/batch_extract.py")
    print("  3. Check data/processed/features.csv")
    print("  4. Open notebooks/01_data_exploration.ipynb")


if __name__ == '__main__':
    main()
