"""Audio processing utilities for speech recognition."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np


_MIN_AUDIO_SECONDS = 0.20
_SILENCE_RMS_THRESHOLD = 0.0025
_FFMPEG_READY = False


class AudioError(Exception):
    code = "audio_error"
    user_message = "I could not process that audio. Please try again."

    def __init__(self, detail: str = ""):
        super().__init__(detail)
        self.detail = detail


class InvalidAudioFormatError(AudioError):
    code = "stt_invalid_format"
    user_message = "That audio format is not supported. Please try again."


class AudioTooShortError(AudioError):
    code = "stt_too_short"
    user_message = "I did not catch that. Hold the button a bit longer and try again."


class NearSilentAudioError(AudioError):
    code = "stt_near_silent"
    user_message = "I could not hear speech clearly. Please speak louder and try again."


class AudioDecodeError(AudioError):
    code = "stt_decode_failed"
    user_message = "I could not decode that audio. Please try again."


class RecognizerFailureError(AudioError):
    code = "stt_recognizer_failed"
    user_message = "I could not transcribe that right now. Please try again."


class RecognizerTimeoutError(AudioError):
    code = "stt_timeout"
    user_message = "Transcription timed out. Please try a shorter request."


def ensure_ffmpeg_available() -> None:
    """Fail fast when ffmpeg is missing so startup catches decoder issues."""
    global _FFMPEG_READY
    if _FFMPEG_READY:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH; install ffmpeg to enable audio decoding")
    _FFMPEG_READY = True


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if not audio_bytes:
        raise InvalidAudioFormatError("empty audio data")
    ensure_ffmpeg_available()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(target_sr),
        "-ac", "1",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, input=audio_bytes, capture_output=True, timeout=30)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found; install ffmpeg to enable audio decoding") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("audio decode timed out") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"failed to decode audio: {stderr[:200]}")

    raw = result.stdout
    if not raw:
        raise AudioDecodeError("ffmpeg produced no output")

    audio_array = np.frombuffer(raw, dtype=np.float32).copy()
    duration_s = len(audio_array) / float(target_sr) if target_sr else 0.0
    if duration_s < _MIN_AUDIO_SECONDS:
        raise AudioTooShortError(f"audio too short: {duration_s:.3f}s")
    rms = float(np.sqrt(np.mean(np.square(audio_array)))) if len(audio_array) else 0.0
    if rms < _SILENCE_RMS_THRESHOLD:
        raise NearSilentAudioError(f"audio near silent: rms={rms:.6f}")
    return audio_array, target_sr
