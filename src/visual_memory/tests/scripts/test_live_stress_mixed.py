"""Mixed HTTP and WebSocket live stress checks."""
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
from visual_memory.tests.scripts.test_websocket_e2e import WsRecorder, _load_audio_case, _load_dataset

_BASE_URL = os.environ.get("TEST_BASE_URL", "").rstrip("/")
_API_KEY = os.environ.get("API_KEY", "")
_CLIENTS = int(os.environ.get("STRESS_MIXED_CLIENTS", "3"))
_DURATION_SECONDS = int(os.environ.get("STRESS_MIXED_DURATION_SECONDS", "60"))
_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUN_DIR = suite_run_dir("live_stress_mixed")
_runner = TestRunner("live_stress_mixed")


def _skip(message: str) -> int:
    print(f"[SKIP] live_stress_mixed: {message}")
    return 0


def _load_assets() -> dict[str, bytes]:
    dataset_path = Path(os.environ.get("WS_E2E_DATASET", "src/visual_memory/tests/input_data/voice_eval_dataset.json"))
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    case_map = _load_dataset(dataset_path)
    teach_image = (_REPO_ROOT / "src/visual_memory/tests/input_images/wallet_1ft_table.jpg").read_bytes()
    scan_image = (_REPO_ROOT / "src/visual_memory/tests/input_images/wallet_3ft_table.jpg").read_bytes()
    ask_audio = _load_audio_case(case_map, os.environ.get("WS_E2E_AUDIO_ASK", "v001"))
    return {
        "teach_image": teach_image,
        "scan_image": scan_image,
        "ask_audio": ask_audio,
    }


def _http_ops(client_id: int, assets: dict[str, bytes]) -> list[dict]:
    client = _RemoteClient(_BASE_URL, _API_KEY)
    ops = []
    started = time.monotonic()
    resp = client.get("/settings")
    ops.append({"op": "settings", "status_code": resp.status_code, "elapsed_ms": round((time.monotonic() - started) * 1000, 2)})

    started = time.monotonic()
    resp = client.post(
        "/scan",
        data={"image": (io.BytesIO(assets["scan_image"]), f"scan-{client_id}.jpg"), "focal_length_px": "3094.0"},
        content_type="multipart/form-data",
    )
    ops.append({"op": "scan", "status_code": resp.status_code, "elapsed_ms": round((time.monotonic() - started) * 1000, 2)})

    started = time.monotonic()
    resp = client.post(
        "/remember",
        data={"image": (io.BytesIO(assets["teach_image"]), f"teach-{client_id}.jpg"), "prompt": f"mixed stress {client_id}"},
        content_type="multipart/form-data",
    )
    ops.append({"op": "remember", "status_code": resp.status_code, "elapsed_ms": round((time.monotonic() - started) * 1000, 2)})
    return ops


def _ws_ops(assets: dict[str, bytes]) -> list[dict]:
    ws = WsRecorder(_BASE_URL, _API_KEY, connect_timeout=12.0)
    out: list[dict] = []
    try:
        t0 = time.monotonic()
        ws.connect()
        out.append({"op": "ws_connect", "status_code": 200, "elapsed_ms": round((time.monotonic() - t0) * 1000, 2)})
        assert ws.wait_event("tts", 0, 15.0) is not None, "missing websocket welcome"
        marker = ws.mark()
        t1 = time.monotonic()
        ws.emit_audio(assets["ask_audio"])
        first = ws.wait_event("transcription", marker, 45.0) or ws.wait_event("error", marker, 45.0)
        assert first is not None, "websocket audio turn produced no response"
        out.append(
            {
                "op": "ws_audio",
                "status_code": 200 if first["name"] != "error" else 500,
                "elapsed_ms": round((time.monotonic() - t1) * 1000, 2),
                "event": first["name"],
            }
        )
    finally:
        ws.disconnect()
    return out


def _client_loop(client_id: int, assets: dict[str, bytes]) -> list[dict]:
    deadline = time.monotonic() + _DURATION_SECONDS
    results: list[dict] = []
    while time.monotonic() < deadline:
        results.extend(_http_ops(client_id, assets))
        results.extend(_ws_ops(assets))
    return results


def test_live_mixed_stress() -> None:
    assert _BASE_URL, "TEST_BASE_URL is required"
    assert _API_KEY, "API_KEY is required"
    assets = _load_assets()
    all_results: list[dict] = []
    failures: list[dict] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=_CLIENTS) as pool:
        futures = [pool.submit(_client_loop, idx, assets) for idx in range(_CLIENTS)]
        for future in as_completed(futures):
            items = future.result()
            with lock:
                all_results.extend(items)
                for item in items:
                    if item["status_code"] >= 500 or item["status_code"] in {401, 403}:
                        failures.append(item)

    write_json(
        _RUN_DIR,
        "results.json",
        {
            "base_url": _BASE_URL,
            "clients": _CLIENTS,
            "duration_seconds": _DURATION_SECONDS,
            "status_counts": summarize_statuses(all_results),
            "p50_ms": round(percentile([r["elapsed_ms"] for r in all_results], 0.50), 2),
            "p95_ms": round(percentile([r["elapsed_ms"] for r in all_results], 0.95), 2),
            "results": all_results,
        },
    )
    append_recommendations(
        [
            "[live_stress_mixed]",
            "- Use STRESS_MIXED_CLIENTS=2 for nominal validation and 3 for realistic developer max.",
            "- Increase STRESS_MIXED_DURATION_SECONDS to 900 or 1800 for soak tiers once the short run is stable.",
        ],
        _RUN_DIR,
    )
    if failures:
        record_failure_modes(failures, _RUN_DIR)
    assert not failures, failures


if __name__ == "__main__":
    if not _BASE_URL or not _API_KEY:
        raise SystemExit(_skip("set TEST_BASE_URL and API_KEY to run mixed live stress."))
    _runner.run("live_stress_mixed:http_and_websocket_soak", test_live_mixed_stress)
    raise SystemExit(_runner.summary())
