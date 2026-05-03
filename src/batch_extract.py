"""
Musical DNA — Batch Feature Extraction
=======================================
Processes all MIDI files in the data/midi/ directory, extracts features,
and saves results to a CSV file.

Usage:
    python src/batch_extract.py

The script expects MIDI files organized in folders by composer:
    data/midi/bach/*.mid
    data/midi/mozart/*.mid
    ...

Output:
    data/processed/features.csv
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import extract_features_from_file, FEATURE_NAMES


def batch_extract(midi_dir='data/midi', output_path='data/processed/features.csv'):
    """Extract features from all MIDI files and save to CSV.
    
    Args:
        midi_dir: Directory containing composer subdirectories
        output_path: Where to save the CSV
    """
    rows = []
    errors = []
    
    midi_path = Path(midi_dir)
    if not midi_path.exists():
        print(f"Error: {midi_dir} not found. Create the directory and add MIDI files.")
        return
    
    # Find all composer directories
    composers = sorted([d.name for d in midi_path.iterdir() if d.is_dir()])
    if not composers:
        print(f"No composer directories found in {midi_dir}/")
        print("Expected structure: data/midi/bach/, data/midi/mozart/, etc.")
        return
    
    print(f"Found {len(composers)} composers: {', '.join(composers)}")
    print("=" * 60)
    
    for composer in composers:
        composer_dir = midi_path / composer
        midi_files = list(composer_dir.glob('*.mid')) + list(composer_dir.glob('*.midi'))
        
        print(f"\n{composer.title()}: {len(midi_files)} files")
        
        for filepath in tqdm(midi_files, desc=f"  Processing {composer}"):
            features = extract_features_from_file(str(filepath))
            
            if features is None:
                errors.append(str(filepath))
                continue
            
            # Add metadata
            features['composer'] = composer
            features['filename'] = filepath.name
            features['filepath'] = str(filepath)
            rows.append(features)
    
    if not rows:
        print("\nNo files were successfully processed!")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Reorder columns: metadata first, then features
    meta_cols = ['composer', 'filename', 'filepath']
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    df = df[meta_cols + feature_cols]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS")
    print(f"  Successfully processed: {len(rows)} files")
    print(f"  Errors (skipped):       {len(errors)} files")
    print(f"  Output saved to:        {output_path}")
    print(f"\nPer-composer counts:")
    for composer, count in df['composer'].value_counts().sort_index().items():
        print(f"  {composer.title():20s} {count:4d} pieces")
    
    if errors:
        print(f"\nFailed files:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    return df


if __name__ == '__main__':
    batch_extract()
