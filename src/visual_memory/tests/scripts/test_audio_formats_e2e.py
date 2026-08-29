"""
End-to-end audio format test.

Sends real audio files to POST /transcribe and verifies Whisper returns text.
Requires the core API to be running (default: http://127.0.0.1:5000).

Usage:
    python -m visual_memory.tests.scripts.test_audio_formats_e2e
    python -m visual_memory.tests.scripts.test_audio_formats_e2e --audio-dir /path/to/audio
    TEST_BASE_URL=http://server:5000 python -m visual_memory.tests.scripts.test_audio_formats_e2e
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

_DEFAULT_AUDIO_DIR = Path(__file__).resolve().parents[1] / "input_audio"
_BASE_URL = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
_API_KEY = os.environ.get("API_KEY", "")

_CONTENT_TYPES = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def _headers() -> dict:
    if _API_KEY:
        return {"X-API-Key": _API_KEY}
    return {}


def run(audio_dir: Path) -> int:
    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in _CONTENT_TYPES
    )
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        print("Expected formats: " + ", ".join(_CONTENT_TYPES))
        return 1

    passed = 0
    failed = 0

    print(f"Testing {len(audio_files)} audio file(s) against {_BASE_URL}/transcribe\n")

    for path in audio_files:
        ctype = _CONTENT_TYPES[path.suffix.lower()]
        audio_bytes = path.read_bytes()
        try:
            resp = requests.post(
                f"{_BASE_URL}/transcribe",
                data=audio_bytes,
                headers={"Content-Type": ctype, **_headers()},
                timeout=60,
            )
        except requests.ConnectionError:
            print(f"  SKIP  {path.name}  (API not reachable at {_BASE_URL})")
            failed += 1
            continue

        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "").strip()
            if text:
                print(f"  PASS  {path.name}  ({ctype})  -> \"{text[:80]}\"")
                passed += 1
            else:
                print(f"  FAIL  {path.name}  ({ctype})  200 but empty transcription")
                failed += 1
        else:
            print(f"  FAIL  {path.name}  ({ctype})  HTTP {resp.status_code}: {resp.text[:120]}")
            failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} passed")
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=_DEFAULT_AUDIO_DIR)
    args = parser.parse_args()
    sys.exit(run(args.audio_dir))


if __name__ == "__main__":
    main()
