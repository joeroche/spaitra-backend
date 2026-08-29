"""Direct OCR concurrency stress checks against the OCR microservice."""
from __future__ import annotations

import io
import os
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

_BASE_URL = os.environ.get("OCR_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
_API_KEY = os.environ.get("OCR_API_KEY", os.environ.get("API_KEY", ""))
_LEVELS = [int(x) for x in os.environ.get("OCR_STRESS_LEVELS", "1,2,3,4").split(",") if x.strip()]
_REQUESTS_PER_LEVEL = int(os.environ.get("OCR_STRESS_REQUESTS_PER_LEVEL", "4"))
_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUN_DIR = suite_run_dir("live_stress_ocr")
_runner = TestRunner("live_stress_ocr")


def _skip(message: str) -> int:
    print(f"[SKIP] live_stress_ocr: {message}")
    return 0


def _image_bytes() -> bytes | None:
    raw = os.environ.get("OCR_STRESS_IMAGE", "").strip()
    path = Path(raw) if raw else (_REPO_ROOT / "src/visual_memory/tests/text_demo/typed.jpeg")
    if not path.exists():
        return None
    return path.read_bytes()


def _one_request(client: _RemoteClient, image: bytes) -> dict:
    resp = client.post(
        "/ocr",
        data={"image": (io.BytesIO(image), "ocr.jpg")},
        content_type="multipart/form-data",
    )
    body = resp.get_json() or {}
    return {
        "status_code": resp.status_code,
        "total_ms": float(body.get("_total_ms", 0.0)),
        "error": body.get("error"),
    }


def test_live_ocr_stress() -> None:
    image = _image_bytes()
    assert image is not None, "OCR stress image missing"
    client = _RemoteClient(_BASE_URL, _API_KEY)
    results_by_level: list[dict] = []
    failures: list[dict] = []

    for level in _LEVELS:
        level_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [pool.submit(_one_request, client, image) for _ in range(_REQUESTS_PER_LEVEL)]
            for future in as_completed(futures):
                item = future.result()
                item["concurrency"] = level
                level_results.append(item)

        statuses = summarize_statuses(level_results)
        p50 = round(percentile([r["total_ms"] for r in level_results if r["total_ms"] > 0], 0.50), 2)
        p95 = round(percentile([r["total_ms"] for r in level_results if r["total_ms"] > 0], 0.95), 2)
        results_by_level.append(
            {
                "concurrency": level,
                "status_counts": statuses,
                "p50_ms": p50,
                "p95_ms": p95,
                "results": level_results,
            }
        )

        for item in level_results:
            code = item["status_code"]
            if code >= 500:
                failures.append(item)
        if not any(item["status_code"] == 200 for item in level_results):
            failures.append(
                {
                    "status_code": 0,
                    "error": f"no_successful_ocr_at_level_{level}",
                    "concurrency": level,
                }
            )

    write_json(
        _RUN_DIR,
        "results.json",
        {
            "base_url": _BASE_URL,
            "requests_per_level": _REQUESTS_PER_LEVEL,
            "levels": results_by_level,
        },
    )
    append_recommendations(
        [
            "[live_stress_ocr]",
            "- Treat concurrency 2 as the shipping target until higher levels show equal stability and p95.",
            "- Above concurrency 2, only explicit 429 overload responses are acceptable.",
        ],
        _RUN_DIR,
    )
    if failures:
        record_failure_modes(failures, _RUN_DIR)
    assert not failures, failures


if __name__ == "__main__":
    image = _image_bytes()
    if image is None:
        raise SystemExit(_skip("missing OCR image; set OCR_STRESS_IMAGE or restore typed.jpeg."))
    _runner.run("live_stress_ocr:concurrency_matrix", test_live_ocr_stress)
    raise SystemExit(_runner.summary())
