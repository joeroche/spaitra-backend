"""
Unit tests for audio decoding helpers.
"""
from __future__ import annotations

import sys

import numpy as np

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.utils import audio_utils as _au
from visual_memory.utils.audio_utils import AudioDecodeError, InvalidAudioFormatError

_runner = TestRunner("audio_utils")


def _reset_ffmpeg_cache():
    _au._FFMPEG_READY = False


def test_ensure_ffmpeg_available_missing_binary():
    old_which = _au.shutil.which
    _reset_ffmpeg_cache()
    _au.shutil.which = lambda _: None
    try:
        try:
            _au.ensure_ffmpeg_available()
        except RuntimeError as exc:
            assert "ffmpeg" in str(exc).lower()
            return
        raise AssertionError("expected RuntimeError when ffmpeg is missing")
    finally:
        _au.shutil.which = old_which
        _reset_ffmpeg_cache()


def test_load_audio_bytes_empty_input():
    try:
        _au.load_audio_bytes(b"")
    except InvalidAudioFormatError:
        return
    raise AssertionError("expected InvalidAudioFormatError for empty payload")


def test_load_audio_bytes_success_path():
    old_which = _au.shutil.which
    old_run = _au.subprocess.run
    _reset_ffmpeg_cache()

    sample = np.full(16000, 0.1, dtype=np.float32).tobytes()

    class _Result:
        returncode = 0
        stdout = sample
        stderr = b""

    _au.shutil.which = lambda _: "/usr/bin/ffmpeg"
    _au.subprocess.run = lambda *args, **kwargs: _Result()
    try:
        audio, sr = _au.load_audio_bytes(b"fake-audio", target_sr=16000)
        assert sr == 16000
        assert len(audio) == 16000
        assert np.isclose(float(np.max(audio)), 0.1)
    finally:
        _au.shutil.which = old_which
        _au.subprocess.run = old_run
        _reset_ffmpeg_cache()


def test_load_audio_bytes_timeout_maps_error():
    old_which = _au.shutil.which
    old_run = _au.subprocess.run
    _reset_ffmpeg_cache()

    def _raise_timeout(*args, **kwargs):
        raise _au.subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

    _au.shutil.which = lambda _: "/usr/bin/ffmpeg"
    _au.subprocess.run = _raise_timeout
    try:
        try:
            _au.load_audio_bytes(b"fake-audio", target_sr=16000)
        except AudioDecodeError as exc:
            assert "timed out" in str(exc).lower()
            return
        raise AssertionError("expected AudioDecodeError on timeout")
    finally:
        _au.shutil.which = old_which
        _au.subprocess.run = old_run
        _reset_ffmpeg_cache()


def test_load_audio_bytes_ffmpeg_exit_error_maps_decode_error():
    old_which = _au.shutil.which
    old_run = _au.subprocess.run
    _reset_ffmpeg_cache()

    class _Result:
        returncode = 1
        stdout = b""
        stderr = b"Invalid data found when processing input"

    _au.shutil.which = lambda _: "/usr/bin/ffmpeg"
    _au.subprocess.run = lambda *args, **kwargs: _Result()
    try:
        try:
            _au.load_audio_bytes(b"fake-audio", target_sr=16000)
        except AudioDecodeError as exc:
            assert "failed to decode audio" in str(exc).lower()
            return
        raise AssertionError("expected AudioDecodeError on ffmpeg non-zero exit")
    finally:
        _au.shutil.which = old_which
        _au.subprocess.run = old_run
        _reset_ffmpeg_cache()


for name, fn in [
    ("audio_utils:ffmpeg_missing", test_ensure_ffmpeg_available_missing_binary),
    ("audio_utils:empty_input", test_load_audio_bytes_empty_input),
    ("audio_utils:success", test_load_audio_bytes_success_path),
    ("audio_utils:timeout", test_load_audio_bytes_timeout_maps_error),
    ("audio_utils:ffmpeg_exit_error", test_load_audio_bytes_ffmpeg_exit_error_maps_decode_error),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
