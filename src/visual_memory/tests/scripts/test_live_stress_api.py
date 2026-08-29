"""Concurrent live API stress checks against the public Spaitra host."""
from __future__ import annotations

import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from visual_memory.tests.scripts.stress_artifacts import (
    append_recommendations,
    percentile,
    record_failure_modes,
    suite_run_dir,
    summarize_statuses,
    write_json,
)
from visual_memory.tests.scripts.test_harness import _RemoteClient, TestRunner

_BASE_URL = os.environ.get("TEST_BASE_URL", "").rstrip("/")
_API_KEY = os.environ.get("API_KEY", "")
_ALLOW_WRITE = os.environ.get("STRESS_ALLOW_WRITE", "1") != "0"
_CONCURRENCY = int(os.environ.get("STRESS_API_CONCURRENCY", "2"))
_ROUNDS = int(os.environ.get("STRESS_API_ROUNDS", "5"))
_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUN_DIR = suite_run_dir("live_stress_api")
_runner = TestRunner("live_stress_api")


def _skip(message: str) -> int:
    print(f"[SKIP] live_stress_api: {message}")
    return 0


def _image_bytes(path_env: str, fallback: str) -> bytes | None:
    raw = os.environ.get(path_env, "").strip()
    path = Path(raw) if raw else (_REPO_ROOT / fallback)
    if not path.exists():
        return None
    return path.read_bytes()


def _audio_payload() -> tuple[bytes, str] | None:
    audio_dir = _REPO_ROOT / "src/visual_memory/tests/input_audio"
    if not audio_dir.exists():
        return None
    for path in sorted(audio_dir.iterdir()):
        if path.suffix.lower() in {".m4a", ".mp3", ".wav", ".webm", ".ogg", ".flac"}:
            ctype = {
                ".m4a": "audio/mp4",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".webm": "audio/webm",
                ".ogg": "audio/ogg",
                ".flac": "audio/flac",
            }[path.suffix.lower()]
            return path.read_bytes(), ctype
    return None


def _task_specs() -> list[dict]:
    tasks = [
        {"name": "health", "method": "get", "path": "/health", "auth": False},
        {"name": "settings", "method": "get", "path": "/settings", "auth": True},
        {"name": "items", "method": "get", "path": "/items", "auth": True},
    ]
    image = _image_bytes("STRESS_SCAN_IMAGE", "src/visual_memory/tests/input_images/wallet_3ft_table.jpg")
    if image is not None:
        tasks.append(
            {
                "name": "scan",
                "method": "post",
                "path": "/scan",
                "auth": True,
                "multipart": {"image": (io.BytesIO(image), "scan.jpg"), "focal_length_px": "3094.0"},
            }
        )
    if _ALLOW_WRITE and image is not None:
        tasks.append(
            {
                "name": "remember",
                "method": "post",
                "path": "/remember",
                "auth": True,
                "multipart": {"image": (io.BytesIO(image), "remember.jpg"), "prompt": "runpod stress item"},
            }
        )
    audio = _audio_payload()
    if audio is not None:
        audio_bytes, content_type = audio
        tasks.append(
            {
                "name": "transcribe",
                "method": "post",
                "path": "/transcribe",
                "auth": True,
                "body": audio_bytes,
                "content_type": content_type,
            }
        )
    return tasks


def _run_task(task: dict, index: int) -> dict:
    client = _RemoteClient(_BASE_URL, _API_KEY if task.get("auth") else "")
    started = time.monotonic()
    if task["method"] == "get":
        resp = client.get(task["path"])
    elif "multipart" in task:
        data = {}
        for key, value in task["multipart"].items():
            if isinstance(value, tuple):
                file_obj, filename = value
                data[key] = (io.BytesIO(file_obj.getvalue()), filename)
            else:
                data[key] = value
        if task["name"] == "remember":
            data["prompt"] = f"runpod stress item {index}"
        resp = client.post(task["path"], data=data, content_type="multipart/form-data")
    else:
        resp = client.post(task["path"], data=task["body"], content_type=task["content_type"])
    elapsed_ms = (time.monotonic() - started) * 1000
    return {
        "task": task["name"],
        "status_code": resp.status_code,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def test_live_api_stress() -> None:
    assert _BASE_URL, "TEST_BASE_URL is required"
    assert _API_KEY, "API_KEY is required"
    tasks = _task_specs()
    assert tasks, "no API stress tasks resolved"
    results: list[dict] = []
    failures: list[dict] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        futures = []
        for round_idx in range(_ROUNDS):
            for task in tasks:
                futures.append(pool.submit(_run_task, task, round_idx))
        for future in as_completed(futures):
            item = future.result()
            with lock:
                results.append(item)
                if item["status_code"] >= 500 or item["status_code"] in {401, 403}:
                    failures.append(item)

    write_json(
        _RUN_DIR,
        "results.json",
        {
            "base_url": _BASE_URL,
            "concurrency": _CONCURRENCY,
            "rounds": _ROUNDS,
            "status_counts": summarize_statuses(results),
            "p50_ms": round(percentile([r["elapsed_ms"] for r in results], 0.50), 2),
            "p95_ms": round(percentile([r["elapsed_ms"] for r in results], 0.95), 2),
            "results": results,
        },
    )
    append_recommendations(
        [
            "[live_stress_api]",
            "- Watch for 5xx responses first; latency spikes matter after stability is established.",
            "- Keep write-heavy stress gated behind STRESS_ALLOW_WRITE when you want read-only validation.",
        ],
        _RUN_DIR,
    )
    if failures:
        record_failure_modes(failures, _RUN_DIR)
    assert not failures, failures


if __name__ == "__main__":
    if not _BASE_URL or not _API_KEY:
        raise SystemExit(_skip("set TEST_BASE_URL and API_KEY to run live API stress."))
    _runner.run("live_stress_api:concurrent_http_mix", test_live_api_stress)
    raise SystemExit(_runner.summary())
