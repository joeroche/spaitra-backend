from __future__ import annotations

import os
import stat
import sys
import tempfile
import types
from pathlib import Path

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.engine.speech_recognition.whisper_recognizer import WhisperRecognizer
from visual_memory.utils import logger as _logger_mod
from visual_memory.utils import metrics as _metrics

_runner = TestRunner("metrics_logger")


def test_open_group_readable_sets_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "group.log"
        fd = _logger_mod._open_group_readable(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.close(fd)
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o660, oct(mode)


def test_open_crash_log_sets_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "crash.log"
        with _logger_mod._open_crash_log(path) as handle:
            handle.write("hello\n")
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o660, oct(mode)
        assert path.read_text(encoding="utf-8") == "hello\n"


def test_collect_system_metrics_skips_mps_on_linux() -> None:
    old_platform = _metrics.sys.platform
    old_torch = sys.modules.get("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Mps:
        @staticmethod
        def current_allocated_memory():
            raise AssertionError("mps probe should not run on linux")

    fake_torch = types.SimpleNamespace(
        cuda=_Cuda(),
        mps=_Mps(),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
    )

    try:
        _metrics.sys.platform = "linux"
        sys.modules["torch"] = fake_torch
        out = _metrics.collect_system_metrics()
        assert "vram_allocated_mb" not in out
    finally:
        _metrics.sys.platform = old_platform
        if old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old_torch


def test_whisper_cpu_probe_suppression_restores_functions() -> None:
    original_cuda_probe = getattr(sys.modules["torch"].cuda, "is_current_stream_capturing", None)
    import transformers.utils.import_utils as import_utils
    original_import_utils_probe = getattr(import_utils, "is_cuda_stream_capturing", None)

    recognizer = WhisperRecognizer.__new__(WhisperRecognizer)
    recognizer.device = "cpu"

    try:
        if original_cuda_probe is not None:
            sys.modules["torch"].cuda.is_current_stream_capturing = lambda: True
        if original_import_utils_probe is not None:
            import_utils.is_cuda_stream_capturing = lambda: True

        with recognizer._suppress_cuda_stream_probe():
            if original_cuda_probe is not None:
                assert sys.modules["torch"].cuda.is_current_stream_capturing() is False
            if original_import_utils_probe is not None:
                assert import_utils.is_cuda_stream_capturing() is False
    finally:
        if original_cuda_probe is not None:
            sys.modules["torch"].cuda.is_current_stream_capturing = original_cuda_probe
        if original_import_utils_probe is not None:
            import_utils.is_cuda_stream_capturing = original_import_utils_probe

    if original_cuda_probe is not None:
        assert sys.modules["torch"].cuda.is_current_stream_capturing is original_cuda_probe
    if original_import_utils_probe is not None:
        assert import_utils.is_cuda_stream_capturing is original_import_utils_probe


for name, fn in [
    ("metrics_logger:group_readable_log", test_open_group_readable_sets_mode),
    ("metrics_logger:group_readable_crash", test_open_crash_log_sets_mode),
    ("metrics_logger:skip_mps_on_linux", test_collect_system_metrics_skips_mps_on_linux),
    ("metrics_logger:whisper_cpu_probe_suppression", test_whisper_cpu_probe_suppression_restores_functions),
]:
    _runner.run(name, fn)

raise SystemExit(_runner.summary())
