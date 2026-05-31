"""
Musical DNA — Week 4: AI Music Classifier Test
================================================
Tests the trained composer classifier on AI-generated MIDI files
(from Suno, Udio, or any other source).

Usage:
    # Test a single file
    python notebooks/04_ai_music_test.py path/to/file.mid

    # Test all MIDIs in a folder
    python notebooks/04_ai_music_test.py data/midi/ai_generated/

    # Test all MIDIs in a folder with known intended composer labels
    python notebooks/04_ai_music_test.py data/midi/ai_generated/ --labels bach,chopin,mozart

Reads:  models/best_classifier.pkl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import load_model, COMPOSERS
from src.features import extract_features_from_file, FEATURE_NAMES


COMPOSERS_ORDERED = ['bach', 'mozart', 'beethoven', 'chopin', 'debussy', 'rachmaninoff']


def classify_file(filepath, pipeline):
    """Extract features from a MIDI file and return classifier prediction + probabilities."""
    features = extract_features_from_file(filepath)
    if features is None:
        return None, None

    X = np.array([[features.get(f, 0) for f in FEATURE_NAMES]])
    prediction = pipeline.predict(X)[0]
    probabilities = dict(zip(pipeline.classes_, pipeline.predict_proba(X)[0]))
    return prediction, probabilities


def print_result(filepath, prediction, probabilities, intended=None):
    filename = Path(filepath).name
    correct = intended and prediction == intended.lower()

    print(f"\n  File: {filename}")
    if intended:
        status = '✓ CORRECT' if correct else '✗ WRONG'
        print(f"  Intended:  {intended.title():<14}  Predicted: {prediction.title():<14}  {status}")
    else:
        print(f"  Predicted: {prediction.title()}")

    print(f"  Confidence breakdown:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for composer, prob in sorted_probs:
        bar = '█' * int(prob * 30)
        marker = ' ←' if composer == prediction else ''
        print(f"    {composer:<16} {prob:>5.1%}  {bar}{marker}")


def main():
    parser = argparse.ArgumentParser(description='Test classifier on AI-generated MIDI files')
    parser.add_argument('path', help='MIDI file or folder of MIDI files')
    parser.add_argument('--labels', default='',
                        help='Comma-separated intended composer labels (in filename order)')
    args = parser.parse_args()

    print('=' * 60)
    print('  MUSICAL DNA — AI Music Classifier Test')
    print('=' * 60)

    # Collect MIDI files
    target = Path(args.path)
    if target.is_file():
        midi_files = [target]
    elif target.is_dir():
        midi_files = sorted(target.glob('*.mid')) + sorted(target.glob('*.midi'))
    else:
        print(f"Error: {args.path} is not a valid file or directory.")
        sys.exit(1)

    if not midi_files:
        print(f"No .mid/.midi files found at {args.path}")
        sys.exit(1)

    # Parse intended labels if provided
    labels = [l.strip().lower() for l in args.labels.split(',') if l.strip()]
    if labels and len(labels) != len(midi_files):
        print(f"Warning: {len(labels)} labels provided for {len(midi_files)} files — ignoring labels.")
        labels = []

    # Load model
    print(f"\n  Loading classifier...")
    pipeline = load_model()
    print(f"  Model: {type(pipeline.named_steps['clf']).__name__}")
    print(f"  Files to test: {len(midi_files)}")

    # Run classification
    print(f"\n── RESULTS ──────────────────────────────────────────────")

    results = []
    for i, filepath in enumerate(midi_files):
        intended = labels[i] if labels else None
        prediction, probabilities = classify_file(str(filepath), pipeline)

        if prediction is None:
            print(f"\n  File: {filepath.name}  →  FAILED (could not parse MIDI)")
            continue

        print_result(filepath, prediction, probabilities, intended)
        results.append({
            'file': filepath.name,
            'intended': intended,
            'predicted': prediction,
            'correct': (prediction == intended) if intended else None,
            'top_confidence': max(probabilities.values()),
        })

    # Summary
    print(f"\n── SUMMARY ──────────────────────────────────────────────")
    if labels:
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        print(f"  Accuracy: {correct}/{total} ({correct/total:.0%})")
        wrong = [r for r in results if not r['correct']]
        if wrong:
            print(f"  Misclassified:")
            for r in wrong:
                print(f"    {r['file']}: intended {r['intended']}, got {r['predicted']}")
    else:
        print(f"  Classified {len(results)} files")
        from collections import Counter
        pred_counts = Counter(r['predicted'] for r in results)
        print(f"  Prediction distribution:")
        for comp, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {comp:<16} {count} file(s)")

    print(f"\n  Tip: re-run with --labels to score accuracy, e.g.:")
    print(f"  python notebooks/04_ai_music_test.py data/midi/ai_generated/ --labels bach,chopin,mozart")


if __name__ == '__main__':
    main()
