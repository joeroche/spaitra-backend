"""
Integration tests for GET /health and debug endpoints.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["ENABLE_DEPTH"] = "0"

import visual_memory.api.pipelines as _pm
from visual_memory.tests.scripts.test_harness import (
    TestRunner, make_test_app, assert_status, seed_item,
)
from visual_memory.api.routes.health import health_bp
import visual_memory.api.routes.health as _health
from visual_memory.api.routes.debug import debug_bp
from visual_memory.api.routes.items import items_bp

_runner = TestRunner("health_debug")
_LOG_DIR = Path(__file__).resolve().parents[4] / "logs"
_APP_LOG = _LOG_DIR / "app.log"
_IMPORTANT_LOG = _LOG_DIR / "important.log"
_CRASH_LOG = _LOG_DIR / "crash.log"


def _seed(db):
    seed_item(db, "wallet", emb_seed=1)
    seed_item(db, "keys", emb_seed=2)


client, db, cleanup = make_test_app([health_bp, debug_bp, items_bp], seed_fn=_seed)


def test_health_ok():
    resp = client.get("/health")
    assert_status(resp, 200)
    data = resp.get_json()
    assert data.get("status") == "ok"


def test_health_dependencies_ok():
    old_ffmpeg = _health.ensure_ffmpeg_available
    old_ocr = _health._ocr_health
    try:
        _health.ensure_ffmpeg_available = lambda: None
        _health._ocr_health = lambda: {
            "enabled": True,
            "reachable": True,
            "latency_ms": 8.2,
            "url": "http://127.0.0.1:8001/health",
        }
        resp = client.get("/health/dependencies")
        assert_status(resp, 200)
        data = resp.get_json()
        assert data.get("status") == "ok"
        assert data["dependencies"]["ffmpeg"]["available"] is True
        assert data["dependencies"]["ocr"]["reachable"] is True
    finally:
        _health.ensure_ffmpeg_available = old_ffmpeg
        _health._ocr_health = old_ocr


def test_health_dependencies_degraded_when_ffmpeg_missing():
    old_ffmpeg = _health.ensure_ffmpeg_available
    old_ocr = _health._ocr_health
    try:
        def _raise():
            raise RuntimeError("ffmpeg not found in PATH")

        _health.ensure_ffmpeg_available = _raise
        _health._ocr_health = lambda: {
            "enabled": False,
            "reachable": None,
            "latency_ms": None,
            "url": "http://127.0.0.1:8001/ocr",
        }
        resp = client.get("/health/dependencies")
        assert_status(resp, 200)
        data = resp.get_json()
        assert data.get("status") == "degraded"
        assert data["dependencies"]["ffmpeg"]["available"] is False
        assert "ffmpeg" in (data["dependencies"]["ffmpeg"]["error"] or "").lower()
    finally:
        _health.ensure_ffmpeg_available = old_ffmpeg
        _health._ocr_health = old_ocr


def test_debug_wipe_items():
    resp = client.post("/debug/wipe", json={"target": "items", "confirm": True})
    assert_status(resp, 200)
    data = resp.get_json()
    assert "wiped" in data or "ok" in data or data.get("target") == "items"
    items = db.get_all_items()
    assert len(items) == 0


def test_debug_config_patch():
    resp = client.patch("/debug/config", json={"similarity_threshold": 0.5})
    assert_status(resp, 200)
    s = _pm._settings
    assert s.similarity_threshold == 0.5


def test_debug_logs_source_filters():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    app_lines = [
        {
            "ts": "2026-03-29T10:00:00",
            "level": "INFO",
            "module": "x",
            "event": "scan_complete",
            "tag": "perf",
            "duration_ms": 10,
        },
        {
            "ts": "2026-03-29T10:00:01",
            "level": "INFO",
            "module": "x",
            "event": "vram_layout",
            "tag": "vram",
            "offloaded": ["gdino"],
            "duration_ms": 30,
            "ram_used_mb": 100,
            "swap_used_mb": 10,
            "vram_allocated_mb": 20,
            "cpu_temp_c": 55.0,
        },
    ]
    _APP_LOG.write_text("\n".join(json.dumps(r) for r in app_lines) + "\n", encoding="utf-8")
    important_lines = [
        {
            "ts": "2026-03-29T10:00:02",
            "level": "WARNING",
            "module": "x",
            "event": "warn_evt",
            "message": "thermal high",
        },
        {
            "ts": "2026-03-29T10:00:03",
            "level": "ERROR",
            "module": "x",
            "event": "err_evt",
            "message": "ocr failure",
        },
    ]
    _IMPORTANT_LOG.write_text("\n".join(json.dumps(r) for r in important_lines) + "\n", encoding="utf-8")
    _CRASH_LOG.write_text(
        "--- CRASH 2026-03-29T10:00:04 ---\nTraceback line 1\n",
        encoding="utf-8",
    )

    resp = client.get("/debug/logs?source=app&event=vram_layout")
    assert_status(resp, 200)
    data = resp.get_json()
    assert data["source"] == "app"
    assert data["count"] == 1
    assert data["entries"][0]["event"] == "vram_layout"

    resp = client.get("/debug/logs?source=important&level=ERROR")
    assert_status(resp, 200)
    data = resp.get_json()
    assert data["source"] == "important"
    assert data["count"] == 1
    assert data["entries"][0]["level"] == "ERROR"

    resp = client.get("/debug/logs?source=crash")
    assert_status(resp, 200)
    data = resp.get_json()
    assert data["source"] == "crash"
    assert data["count"] == 1


def test_debug_perf_summary():
    resp = client.get("/debug/perf?n=5")
    assert_status(resp, 200)
    data = resp.get_json()
    assert "metrics_now" in data
    assert "summary" in data
    assert "recent_perf_entries" in data
    assert "important_counts" in data
    assert data["summary"]["vram_layout_count"] >= 1
    assert data["summary"]["vram_swap_count"] >= 1
    assert data["important_counts"]["warning"] >= 1
    assert data["important_counts"]["error_or_critical"] >= 1


for name, fn in [
    ("health:ok", test_health_ok),
    ("health:dependencies_ok", test_health_dependencies_ok),
    ("health:dependencies_degraded", test_health_dependencies_degraded_when_ffmpeg_missing),
    ("debug:wipe_items", test_debug_wipe_items),
    ("debug:config_patch", test_debug_config_patch),
    ("debug:logs_source_filters", test_debug_logs_source_filters),
    ("debug:perf_summary", test_debug_perf_summary),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
