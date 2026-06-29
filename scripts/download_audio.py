"""
Download YouTube audio for a copyright case pair, ready for MIDI conversion.

Usage:
    python scripts/download_audio.py <case_slug> plaintiff "<YouTube URL>"
    python scripts/download_audio.py <case_slug> defendant "<YouTube URL>"

Example:
    python scripts/download_audio.py williams_v_gaye plaintiff "https://youtube.com/..."
    python scripts/download_audio.py williams_v_gaye defendant "https://youtube.com/..."

After downloading:
    1. Go to basicpitch.spotify.com
    2. Upload the .mp3 file
    3. Click Download MIDI
    4. Save it to the same folder as plaintiff.mid or defendant.mid
    5. Run: python scripts/link_midis.py

Audio files land in data/midi/copyright_cases/<case_slug>/
as plaintiff.mp3 or defendant.mp3.
"""

import sys
import subprocess
from pathlib import Path

MIDI_ROOT = Path('data/midi/copyright_cases')


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    case_slug, role, url = sys.argv[1], sys.argv[2], sys.argv[3]

    if role not in ('plaintiff', 'defendant'):
        print(f"  ERROR: role must be 'plaintiff' or 'defendant', got '{role}'")
        sys.exit(1)

    case_dir = MIDI_ROOT / case_slug
    if not case_dir.exists():
        print(f"  ERROR: case folder not found: {case_dir}")
        print(f"  Available: {[d.name for d in MIDI_ROOT.iterdir() if d.is_dir()][:5]} ...")
        sys.exit(1)

    out_path = case_dir / role  # yt-dlp adds the extension

    print(f"  Case:  {case_slug}")
    print(f"  Role:  {role}")
    print(f"  URL:   {url}")
    print(f"  Out:   {out_path}.mp3")
    print()

    cmd = [
        'yt-dlp',
        '--js-runtimes', 'node',       # required on macOS
        '-x',                          # extract audio only
        '--audio-format', 'mp3',
        '--audio-quality', '0',        # best quality
        '-o', str(out_path) + '.%(ext)s',
        '--no-playlist',
        url,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print('\n  ERROR: yt-dlp failed. Check the URL and try again.')
        sys.exit(1)

    mp3 = out_path.with_suffix('.mp3')
    if mp3.exists():
        size_kb = mp3.stat().st_size // 1024
        print(f'\n  Downloaded: {mp3}  ({size_kb} KB)')
        print(f'\n  Next step:')
        print(f'    1. Go to  basicpitch.spotify.com')
        print(f'    2. Upload {mp3.name}')
        print(f'    3. Download the MIDI and save as:')
        print(f'       {case_dir}/{role}.mid')
        print(f'    4. Run: python scripts/link_midis.py')
    else:
        print(f'\n  WARNING: expected {mp3} not found — check yt-dlp output above')


if __name__ == '__main__':
    main()
