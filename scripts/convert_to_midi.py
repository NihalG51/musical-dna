"""
Automate basicpitch.spotify.com to convert MP3 files to MIDI.

Scans all case folders for plaintiff.mp3 / defendant.mp3 that don't yet
have a matching .mid file, uploads each to Basic Pitch, and saves the
downloaded MIDI alongside the MP3.

Usage:
    python scripts/convert_to_midi.py              # process all pending
    python scripts/convert_to_midi.py --case williams_v_gaye
    python scripts/convert_to_midi.py --headless   # no visible browser
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

MIDI_ROOT  = Path('data/midi/copyright_cases')
BASIC_PITCH = 'https://basicpitch.spotify.com'
PROCESS_TIMEOUT = 600_000   # 10 min max per file
POLL_INTERVAL   = 2_000     # check for download button every 2s
TRIM_SECONDS    = 90        # upload only the first 90s — contested passages
                            # are almost always in the intro/first chorus


def pending_conversions(case_slug=None):
    """Return list of (mp3_path, midi_path) pairs that still need conversion."""
    pairs = []
    dirs = [MIDI_ROOT / case_slug] if case_slug else sorted(MIDI_ROOT.iterdir())
    for case_dir in dirs:
        if not case_dir.is_dir():
            continue
        for role in ('plaintiff', 'defendant'):
            mp3  = case_dir / f'{role}.mp3'
            midi = case_dir / f'{role}.mid'
            if mp3.exists() and not midi.exists():
                pairs.append((mp3, midi))
    return pairs


def trim_audio(mp3_path: Path, seconds: int) -> Path:
    """Use ffmpeg to trim mp3 to first N seconds, returns path to temp file."""
    tmp = Path(tempfile.mktemp(suffix='.mp3'))
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(mp3_path),
         '-t', str(seconds), '-acodec', 'copy', str(tmp)],
        check=True, capture_output=True,
    )
    return tmp


def convert_one(page, mp3_path: Path, midi_path: Path):
    print(f'\n  Converting: {mp3_path}')

    # Trim to first TRIM_SECONDS — faster processing, focuses on contested passage
    trimmed = trim_audio(mp3_path, TRIM_SECONDS)
    print(f'    Trimmed to {TRIM_SECONDS}s for upload.')

    # Fresh page load each time to reset state
    page.goto(BASIC_PITCH, wait_until='networkidle', timeout=30_000)

    # Upload the file via the hidden <input type="file">
    file_input = page.locator('input[type="file"]')
    file_input.set_input_files(str(trimmed))
    print(f'    Uploaded. Waiting for Basic Pitch to process...')

    # Wait for a download link to appear — Basic Pitch adds a
    # "Download MIDI" button once the in-browser model finishes.
    # We poll for any <a> or <button> whose text contains "midi" or "download".
    download_locator = page.locator(
        'a:has-text("MIDI"), a:has-text("midi"), '
        'button:has-text("MIDI"), button:has-text("midi"), '
        'a:has-text("Download"), button:has-text("Download")'
    ).first

    start = time.time()
    while True:
        try:
            download_locator.wait_for(state='visible', timeout=POLL_INTERVAL)
            break
        except PWTimeout:
            elapsed = int(time.time() - start)
            if elapsed * 1000 >= PROCESS_TIMEOUT:
                raise RuntimeError(f'Timed out after {elapsed}s waiting for download button')
            print(f'    Still processing... ({elapsed}s elapsed)')

    print(f'    Done processing. Downloading MIDI...')

    # Intercept the download
    with page.expect_download(timeout=30_000) as dl_info:
        download_locator.click()

    download = dl_info.value
    download.save_as(str(midi_path))
    trimmed.unlink(missing_ok=True)
    size_kb = midi_path.stat().st_size // 1024
    print(f'    Saved: {midi_path}  ({size_kb} KB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',     default=None, help='Process only this case slug')
    parser.add_argument('--headless', action='store_true', help='Run without visible browser')
    args = parser.parse_args()

    pairs = pending_conversions(args.case)
    if not pairs:
        print('  Nothing to convert — all MP3s already have a matching .mid file.')
        return

    print(f'  Found {len(pairs)} file(s) to convert:')
    for mp3, midi in pairs:
        print(f'    {mp3}  →  {midi.name}')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(accept_downloads=True)
        page    = context.new_page()

        ok, failed = [], []
        for mp3_path, midi_path in pairs:
            try:
                convert_one(page, mp3_path, midi_path)
                ok.append(mp3_path)
            except Exception as e:
                print(f'    ERROR: {e}')
                # Screenshot for debugging
                shot = Path('/private/tmp') / f'basicpitch_error_{mp3_path.stem}.png'
                page.screenshot(path=str(shot))
                print(f'    Screenshot: {shot}')
                failed.append(mp3_path)

        browser.close()

    print(f'\n  ── Summary ──────────────────────────────')
    print(f'  Converted:  {len(ok)}')
    if failed:
        print(f'  Failed:     {len(failed)}')
        for f in failed:
            print(f'    {f}')
    print(f'\n  Run next:  python scripts/link_midis.py')
    print(f'             python notebooks/05_score_pair.py --auto-score-all')


if __name__ == '__main__':
    main()
