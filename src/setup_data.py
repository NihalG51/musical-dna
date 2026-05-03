"""
Musical DNA — MIDI Data Setup
==============================
Downloads and organizes MIDI files from music21's built-in corpus.
This gives you a starter dataset while you gather additional files.

Usage:
    python src/setup_data.py

NOTE: The music21 corpus has limited pieces for some composers.
After running this, you'll need to download additional MIDI files
manually from IMSLP, MuseScore, etc. (see instructions at the end).
"""

import os
import sys
import signal
import warnings
from pathlib import Path

# Suppress music21 warnings about overfull measures
warnings.filterwarnings('ignore', category=UserWarning)

from music21 import corpus, converter


# Timeout handler for pieces that take too long to parse
class ParseTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise ParseTimeoutError("Parsing took too long")


def create_directory_structure():
    """Create the data directory structure."""
    composers = ['bach', 'mozart', 'beethoven', 'chopin', 'debussy', 'rachmaninoff']
    
    for composer in composers:
        path = Path(f'data/midi/{composer}')
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {path}/")
    
    for d in ['data/processed', 'data/copyright_cases', 'models', 'notebooks', 'dashboard']:
        Path(d).mkdir(parents=True, exist_ok=True)
        gitkeep = Path(d) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()


def safe_export(corpus_path, out_path, timeout_sec=30):
    """Parse and export a single piece with a timeout.
    
    Returns True on success, False on failure.
    """
    try:
        # Set a timeout (Unix/Mac only — on Windows this is skipped)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)
        
        score = corpus.parse(corpus_path)
        score.write('midi', fp=out_path)
        
        # Cancel the alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        return True
    except ParseTimeoutError:
        print(f"    Skipped (too slow): {Path(str(corpus_path)).stem}")
        return False
    except Exception as e:
        return False
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def export_corpus_pieces():
    """Export pieces from music21's built-in corpus as MIDI files."""
    
    total = 0
    
    # ---- BACH ----
    # Bach chorales are small and parse quickly
    print("\nBach — exporting chorales (these are fast)...")
    bach_works = corpus.getComposer('bach')
    count = 0
    for path in sorted(bach_works)[:50]:
        filename = Path(str(path)).stem.replace(' ', '_')
        out_path = f'data/midi/bach/{filename}.mid'
        if not os.path.exists(out_path):
            if safe_export(path, out_path, timeout_sec=15):
                count += 1
                print(f"    [{count}] {filename}")
    print(f"  Exported {count} Bach pieces")
    total += count
    
    # ---- MOZART ----
    print("\nMozart — exporting available pieces...")
    mozart_works = corpus.getComposer('mozart')
    count = 0
    for path in sorted(mozart_works):
        filename = Path(str(path)).stem.replace(' ', '_')
        parent = Path(str(path)).parent.stem
        safe_name = f"{parent}_{filename}"
        out_path = f'data/midi/mozart/{safe_name}.mid'
        if not os.path.exists(out_path):
            if safe_export(path, out_path, timeout_sec=30):
                count += 1
                print(f"    [{count}] {safe_name}")
    print(f"  Exported {count} Mozart pieces")
    total += count
    
    # ---- BEETHOVEN ----
    # These are string quartets — large and slow. Only grab a few.
    print("\nBeethoven — exporting (limited, these are large quartet scores)...")
    beethoven_works = corpus.getComposer('beethoven')
    count = 0
    for path in sorted(beethoven_works)[:10]:
        filename = Path(str(path)).stem.replace(' ', '_')
        parent = Path(str(path)).parent.stem
        safe_name = f"{parent}_{filename}"
        out_path = f'data/midi/beethoven/{safe_name}.mid'
        if not os.path.exists(out_path):
            if safe_export(path, out_path, timeout_sec=45):
                count += 1
                print(f"    [{count}] {safe_name}")
    print(f"  Exported {count} Beethoven pieces")
    total += count
    
    return total


def main():
    print("=" * 60)
    print("  MUSICAL DNA — Data Setup")
    print("=" * 60)
    
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    print("\n2. Exporting pieces from music21 corpus...")
    print("   (This should take 2-5 minutes)")
    total = export_corpus_pieces()
    
    print("\n" + "=" * 60)
    print(f"  Done! Exported {total} pieces from the built-in corpus.")
    print("=" * 60)
    
    print("""
  The music21 corpus gives you a HEAD START, but it's not enough.
  You need 30-50 piano MIDI files per composer. Here's where to
  get the rest:

  PRIORITY DOWNLOADS (do this today):

  Chopin (you have 0 — need 30-40):
    musescore.com — search "Chopin" and export as MIDI
    imslp.org/wiki/Category:Chopin,_Frederic
    Good pieces: Nocturnes, Preludes, Etudes, Waltzes
    Save to: data/midi/chopin/
  
  Debussy (you have 0 — need 25-35):
    musescore.com — search "Debussy piano"
    imslp.org/wiki/Category:Debussy,_Claude
    Good pieces: Preludes, Arabesques, Suite bergamasque
    Save to: data/midi/debussy/
  
  Rachmaninoff (you have 0 — need 20-30):
    musescore.com — search "Rachmaninoff"
    imslp.org/wiki/Category:Rachmaninoff,_Sergei
    Good pieces: Preludes, Etudes-Tableaux, Moments Musicaux
    Save to: data/midi/rachmaninoff/
  
  MORE Bach/Mozart/Beethoven piano works:
    Search for piano sonatas, variations, and solo keyboard works
    The corpus mostly has chorales (Bach) and quartets (Beethoven)
    You want PIANO works for consistency

  IMPORTANT: Only download piano/solo keyboard pieces for now.
  Orchestral MIDI files have inconsistent instrument mapping.

  Once you have 30+ files per composer, run:
    python src/batch_extract.py
""")


if __name__ == '__main__':
    main()