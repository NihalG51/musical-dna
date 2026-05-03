"""
Musical DNA — MAESTRO Dataset Importer
=======================================
Sorts MAESTRO MIDI files into your data/midi/<composer>/ folders
using the metadata CSV.

Usage:
    1. Download maestro-v3.0.0-midi.zip from:
       https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
    
    2. Download maestro-v3.0.0.csv from:
       https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv

    3. Unzip the midi zip file somewhere (e.g., ~/Downloads/maestro-v3.0.0/)
    
    4. Run this script:
       python src/import_maestro.py ~/Downloads/maestro-v3.0.0 ~/Downloads/maestro-v3.0.0.csv
"""

import os
import sys
import csv
import shutil
from pathlib import Path
from collections import Counter


# Map MAESTRO composer names to your folder names
# Only include the 6 composers you're studying
COMPOSER_MAP = {
    'Johann Sebastian Bach': 'bach',
    'Wolfgang Amadeus Mozart': 'mozart',
    'Ludwig van Beethoven': 'beethoven',
    'Frédéric Chopin': 'chopin',
    'Frederic Chopin': 'chopin',
    'Claude Debussy': 'debussy',
    'Sergei Rachmaninoff': 'rachmaninoff',
    'Sergei Rachmaninov': 'rachmaninoff',
}

# How many files to copy per composer (don't need all of them)
MAX_PER_COMPOSER = 50


def import_maestro(maestro_dir, csv_path):
    """Sort MAESTRO MIDI files into data/midi/<composer>/ folders."""
    
    maestro_path = Path(maestro_dir)
    
    # Read the metadata CSV
    print(f"Reading metadata from {csv_path}...")
    entries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    
    print(f"  Found {len(entries)} total performances in MAESTRO")
    
    # Count composers
    all_composers = Counter(e['canonical_composer'] for e in entries)
    print(f"\n  Composers in MAESTRO ({len(all_composers)} total):")
    for composer, count in all_composers.most_common(20):
        marker = " <-- YOUR COMPOSER" if composer in COMPOSER_MAP else ""
        print(f"    {composer:40s} {count:4d} pieces{marker}")
    
    # Filter to your 6 composers
    relevant = [e for e in entries if e['canonical_composer'] in COMPOSER_MAP]
    print(f"\n  Relevant to your project: {len(relevant)} pieces")
    
    # Copy files
    copied = Counter()
    skipped = 0
    errors = 0
    
    print("\nCopying MIDI files...")
    for entry in relevant:
        composer_folder = COMPOSER_MAP[entry['canonical_composer']]
        
        # Check if we already have enough for this composer
        if copied[composer_folder] >= MAX_PER_COMPOSER:
            skipped += 1
            continue
        
        # Find the MIDI file
        midi_rel_path = entry['midi_filename']
        midi_src = maestro_path / midi_rel_path
        
        if not midi_src.exists():
            errors += 1
            continue
        
        # Create a readable filename from the title
        title = entry.get('canonical_title', midi_src.stem)
        # Clean up the title for use as a filename
        safe_title = "".join(c if c.isalnum() or c in ' -_' else '_' for c in title)
        safe_title = safe_title.strip().replace('  ', ' ').replace(' ', '_')
        # Truncate if too long
        if len(safe_title) > 80:
            safe_title = safe_title[:80]
        
        dest_dir = Path(f'data/midi/{composer_folder}')
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_dir / f"maestro_{safe_title}.mid"
        
        # Skip if already exists
        if dest_path.exists():
            continue
        
        # Handle duplicate filenames by adding a number
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"maestro_{safe_title}_{counter}.mid"
            counter += 1
        
        shutil.copy2(midi_src, dest_path)
        copied[composer_folder] += 1
        print(f"  [{copied[composer_folder]:2d}] {composer_folder}/{dest_path.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  MAESTRO Import Complete")
    print("=" * 60)
    print(f"\n  Files copied per composer:")
    for composer in ['bach', 'mozart', 'beethoven', 'chopin', 'debussy', 'rachmaninoff']:
        count = copied.get(composer, 0)
        status = "OK" if count >= 20 else "need more" if count > 0 else "NONE — download from piano-midi.de"
        print(f"    {composer:20s} {count:3d} files  ({status})")
    
    total_copied = sum(copied.values())
    print(f"\n  Total copied: {total_copied}")
    if skipped:
        print(f"  Skipped (already had {MAX_PER_COMPOSER}): {skipped}")
    if errors:
        print(f"  Errors (file not found): {errors}")
    
    # Check what's still needed
    print("\n  Next steps:")
    for composer in ['bach', 'mozart', 'beethoven', 'chopin', 'debussy', 'rachmaninoff']:
        existing = len(list(Path(f'data/midi/{composer}').glob('*.mid')))
        if existing < 25:
            need = 30 - existing
            print(f"    {composer}: need ~{need} more files (download from piano-midi.de)")
    
    print(f"\n  Once you have 25+ files per composer, run:")
    print(f"    python src/batch_extract.py")


def main():
    if len(sys.argv) < 3:
        print("Usage: python src/import_maestro.py <maestro_folder> <maestro_csv>")
        print()
        print("Example:")
        print("  python src/import_maestro.py ~/Downloads/maestro-v3.0.0 ~/Downloads/maestro-v3.0.0.csv")
        print()
        print("Download the files first:")
        print("  MIDI: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip")
        print("  CSV:  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv")
        sys.exit(1)
    
    maestro_dir = sys.argv[1]
    csv_path = sys.argv[2]
    
    if not os.path.isdir(maestro_dir):
        print(f"Error: {maestro_dir} is not a directory")
        print("Make sure you unzipped maestro-v3.0.0-midi.zip first")
        sys.exit(1)
    
    if not os.path.isfile(csv_path):
        print(f"Error: {csv_path} not found")
        print("Download it from: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv")
        sys.exit(1)
    
    import_maestro(maestro_dir, csv_path)


if __name__ == '__main__':
    main()