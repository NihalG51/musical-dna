"""
Scan data/midi/copyright_cases/<case_slug>/ for plaintiff.mid and defendant.mid,
then write the paths into cases.csv so --auto-score-all can find them.

Run this after dropping MIDI files into the case folders:
    python scripts/link_midis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

CASES_CSV = Path('data/copyright_cases/cases.csv')
MIDI_ROOT = Path('data/midi/copyright_cases')


def normalize(s: str) -> str:
    """Strip all non-alphanumeric chars for fuzzy folder matching."""
    import re
    return re.sub(r'[^a-z0-9]', '', s.lower())


def main():
    df = pd.read_csv(CASES_CSV, dtype=str)

    # Build normalized-name → folder-path index for fuzzy matching
    folder_index = {normalize(d.name): d
                    for d in MIDI_ROOT.iterdir() if d.is_dir()}

    updated = 0
    missing = []

    for i, row in df.iterrows():
        case_name = row['case_name']
        key = normalize(case_name)
        case_dir = folder_index.get(key)

        if case_dir is None:
            # Partial prefix match (first 15 chars after normalization)
            candidates = [d for k, d in folder_index.items() if k[:15] == key[:15]]
            case_dir = candidates[0] if len(candidates) == 1 else None

        if case_dir is None:
            missing.append(f"  NO FOLDER  {case_name}")
            continue

        p_path = case_dir / 'plaintiff.mid'
        d_path = case_dir / 'defendant.mid'

        p_exists = p_path.exists()
        d_exists = d_path.exists()

        if p_exists:
            df.at[i, 'midi_plaintiff_path'] = str(p_path)
        if d_exists:
            df.at[i, 'midi_defendant_path'] = str(d_path)

        if p_exists or d_exists:
            updated += 1
            status = []
            if p_exists:
                status.append('plaintiff')
            if d_exists:
                status.append('defendant')
            print(f"  OK  {case_name}  [{', '.join(status)}]")
        else:
            missing.append(f"  NO MIDI    {case_name}  (folder: {case_dir.name}/)")

    df.to_csv(CASES_CSV, index=False)
    print(f"\n  Updated {updated} case(s) in {CASES_CSV}")

    if missing:
        print(f"\n  Cases with no MIDI yet ({len(missing)}):")
        for m in missing:
            print(m)


if __name__ == '__main__':
    main()
