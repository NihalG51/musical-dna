"""
Musical DNA — Week 5: Pairwise MIDI Similarity Scorer
======================================================
Computes all four pairwise similarity scores between two MIDI files.
Use this to score any copyright case where you have MIDI for both works.

Usage:
    python notebooks/05_score_pair.py plaintiff.mid defendant.mid
    python notebooks/05_score_pair.py plaintiff.mid defendant.mid --case "Williams v. Gaye"

To auto-score all cases in cases.csv that have MIDI paths set:
    python notebooks/05_score_pair.py --auto-score-all
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.similarity import compute_pairwise_similarity
from src.cases import load_cases, auto_score, save_cases, validate, print_summary


def score_pair(path1, path2, case_name=''):
    label = f" — {case_name}" if case_name else ''
    print(f"\n  Scoring{label}:")
    print(f"  Plaintiff: {Path(path1).name}")
    print(f"  Defendant: {Path(path2).name}")

    scores = compute_pairwise_similarity(path1, path2)
    if scores is None:
        print("  ERROR: Could not load one or both MIDI files.")
        return

    bar = lambda v: '█' * int(v * 30) + f'  {v:.3f}'

    print(f"\n  {'Metric':<28} Score")
    print('  ' + '-' * 50)
    print(f"  {'Melodic similarity':<28} {bar(scores['melodic_similarity'])}")
    print(f"  {'Harmonic cosine sim':<28} {bar(scores['harmonic_cosine_sim'])}")
    print(f"  {'Rhythmic DTW sim':<28} {bar(scores['rhythmic_dtw_sim'])}")
    print(f"  {'N-gram overlap (n=4)':<28} {bar(scores['ngram_overlap_4'])}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Overall average:             {bar(avg)}")

    print(f"\n  Interpretation guide:")
    if avg >= 0.7:
        print(f"  HIGH similarity — strong case for musical overlap")
    elif avg >= 0.4:
        print(f"  MODERATE similarity — mixed signals, depends on which elements courts focus on")
    else:
        print(f"  LOW similarity — hard to argue substantial similarity on musical grounds")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', help='Two MIDI file paths (plaintiff then defendant)')
    parser.add_argument('--case', default='', help='Case name label')
    parser.add_argument('--auto-score-all', action='store_true',
                        help='Auto-score all rows in cases.csv with MIDI paths set')
    args = parser.parse_args()

    print('=' * 60)
    print('  MUSICAL DNA — Pairwise Similarity Scorer')
    print('=' * 60)

    if args.auto_score_all:
        print('\n  Loading cases.csv...')
        df = load_cases()
        print_summary(df)

        print('\n── AUTO-SCORING (rows with both MIDI paths set) ─────────')
        df = auto_score(df)
        save_cases(df)

        print('\n── VALIDATION ───────────────────────────────────────────')
        issues = validate(df)
        if issues:
            for issue in issues:
                print(issue)
        else:
            print('  All cases look clean.')
        return

    if len(args.files) != 2:
        print('\n  Usage: python notebooks/05_score_pair.py plaintiff.mid defendant.mid')
        print('         python notebooks/05_score_pair.py --auto-score-all')
        sys.exit(1)

    score_pair(args.files[0], args.files[1], args.case)


if __name__ == '__main__':
    main()
