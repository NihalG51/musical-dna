"""
Musical DNA — MIDI Data Setup
==============================
Downloads and organizes MIDI files from music21's built-in corpus
for the initial dataset. This gives you a head start while you
gather additional MIDI files from external sources.

Usage:
    python src/setup_data.py

This creates:
    data/midi/bach/       — Bach chorales and keyboard works
    data/midi/mozart/     — Mozart piano sonatas
    data/midi/beethoven/  — Beethoven piano sonatas
    data/midi/chopin/     — (placeholder — download from external sources)
    data/midi/debussy/    — (placeholder — download from external sources)

After running this, supplement with MIDI files from:
    - Lakh MIDI Dataset (lmd.io)
    - IMSLP.org
    - MuseScore.com
    - KernScores (kern.humdrum.org)
"""

import os
import sys
from pathlib import Path
from music21 import corpus, converter


def create_directory_structure():
    """Create the data directory structure."""
    composers = ['bach', 'mozart', 'beethoven', 'chopin', 'debussy', 'rachmaninoff']
    
    for composer in composers:
        path = Path(f'data/midi/{composer}')
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {path}/")
    
    # Also create processed and copyright_cases dirs
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/copyright_cases').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('notebooks').mkdir(parents=True, exist_ok=True)
    Path('dashboard').mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep files for empty directories
    for d in ['data/processed', 'data/copyright_cases', 'models', 'dashboard']:
        gitkeep = Path(d) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()


def export_corpus_pieces():
    """Export pieces from music21's built-in corpus as MIDI files."""
    
    # Bach — chorales are easily available in the corpus
    print("\nExporting Bach pieces from music21 corpus...")
    bach_chorales = corpus.getComposer('bach')
    exported_bach = 0
    for path in bach_chorales[:50]:  # Get up to 50
        try:
            score = corpus.parse(path)
            filename = Path(path).stem.replace(' ', '_')
            out_path = f'data/midi/bach/{filename}.mid'
            if not os.path.exists(out_path):
                score.write('midi', fp=out_path)
                exported_bach += 1
        except Exception as e:
            continue
    print(f"  Exported {exported_bach} Bach pieces")
    
    # Mozart — K545 and other available pieces
    print("\nExporting Mozart pieces from music21 corpus...")
    mozart_pieces = corpus.getComposer('mozart')
    exported_mozart = 0
    for path in mozart_pieces[:50]:
        try:
            score = corpus.parse(path)
            filename = Path(path).stem.replace(' ', '_')
            out_path = f'data/midi/mozart/{filename}.mid'
            if not os.path.exists(out_path):
                score.write('midi', fp=out_path)
                exported_mozart += 1
        except Exception as e:
            continue
    print(f"  Exported {exported_mozart} Mozart pieces")
    
    # Beethoven
    print("\nExporting Beethoven pieces from music21 corpus...")
    beethoven_pieces = corpus.getComposer('beethoven')
    exported_beethoven = 0
    for path in beethoven_pieces[:50]:
        try:
            score = corpus.parse(path)
            filename = Path(path).stem.replace(' ', '_')
            out_path = f'data/midi/beethoven/{filename}.mid'
            if not os.path.exists(out_path):
                score.write('midi', fp=out_path)
                exported_beethoven += 1
        except Exception as e:
            continue
    print(f"  Exported {exported_beethoven} Beethoven pieces")
    
    return exported_bach + exported_mozart + exported_beethoven


def main():
    print("=" * 60)
    print("  MUSICAL DNA — Data Setup")
    print("=" * 60)
    
    # Step 1: Create directories
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Export from corpus
    print("\n2. Exporting pieces from music21 corpus...")
    total = export_corpus_pieces()
    
    # Step 3: Summary
    print("\n" + "=" * 60)
    print(f"  Setup complete! Exported {total} pieces total.")
    print("=" * 60)
    
    print("\n  Your data/midi/ directory now has starter MIDI files for")
    print("  Bach, Mozart, and Beethoven from the music21 built-in corpus.")
    
    print("\n  TO DO: Download additional MIDI files for all 6 composers")
    print("  from these sources:")
    print("    - Lakh MIDI Dataset: https://colinraffel.com/projects/lmd/")
    print("    - IMSLP:             https://imslp.org/")
    print("    - MuseScore:         https://musescore.com/")
    print("    - KernScores:        https://kern.humdrum.org/")
    
    print(f"\n  Target: 30-50 MIDI files per composer (piano works only)")
    print(f"\n  Especially needed:")
    print(f"    data/midi/chopin/       — Download from IMSLP or MuseScore")
    print(f"    data/midi/debussy/      — Download from IMSLP or MuseScore")
    print(f"    data/midi/rachmaninoff/ — Download from IMSLP or MuseScore")
    
    print(f"\n  Once you have files, run: python src/batch_extract.py")


if __name__ == '__main__':
    main()
