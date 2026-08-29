"""Runtime validation for the RunPod deployment shape."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from visual_memory.tests.scripts.stress_artifacts import suite_run_dir, write_json
from visual_memory.tests.scripts.test_harness import TestRunner

_runner = TestRunner("runpod_runtime")
_RUN_DIR = suite_run_dir("runpod_runtime")
_OPT_ROOT = Path("/opt/spaitra")
_WORKSPACE_ROOT = Path("/workspace/spaitra")
_RUNTIME_ENV_ROOT = Path("/etc/spaitra")
_STRICT = os.environ.get("RUNPOD_RUNTIME_REQUIRED", "0") == "1"


def _run_as_service_user(args: list[str]) -> subprocess.CompletedProcess[str]:
    if os.geteuid() == 0:
        cmd = ["sudo", "-u", "spaitra", *args]
    else:
        cmd = args
    return subprocess.run(cmd, capture_output=True, text=True)


def _skip(message: str) -> int:
    print(f"[SKIP] runpod_runtime: {message}")
    return 0


def _http_json(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read().decode()
            return resp.status, json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode()
        try:
            return exc.code, json.loads(payload) if payload else None
        except Exception:
            return exc.code, None


def _read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def test_paths_and_user() -> None:
    assert _WORKSPACE_ROOT.exists(), f"missing workspace root: {_WORKSPACE_ROOT}"
    assert _OPT_ROOT.exists(), f"missing compat root: {_OPT_ROOT}"
    assert _OPT_ROOT.is_symlink(), "/opt/spaitra must be a symlink"
    assert os.readlink(_OPT_ROOT) == str(_WORKSPACE_ROOT), f"/opt/spaitra -> {os.readlink(_OPT_ROOT)!r}"
    proc = subprocess.run(["id", "spaitra"], capture_output=True, text=True)
    assert proc.returncode == 0, "service user spaitra missing"


def test_env_permissions_and_flags() -> None:
    core_env = _WORKSPACE_ROOT / ".env"
    ocr_env = _WORKSPACE_ROOT / ".ocr.env"
    runtime_core_env = _RUNTIME_ENV_ROOT / ".env"
    runtime_ocr_env = _RUNTIME_ENV_ROOT / ".ocr.env"
    assert core_env.exists(), f"missing env file: {core_env}"
    assert ocr_env.exists(), f"missing env file: {ocr_env}"
    assert runtime_core_env.exists(), f"missing runtime env file: {runtime_core_env}"
    assert runtime_ocr_env.exists(), f"missing runtime env file: {runtime_ocr_env}"
    for path in [core_env, ocr_env]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode in {0o600, 0o666}, f"{path} expected 0600 or RunPod fuse 0666 got {oct(mode)}"
    for path in [runtime_core_env, runtime_ocr_env]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"{path} expected 0600 got {oct(mode)}"
    for path in [runtime_core_env, runtime_ocr_env]:
        proc = _run_as_service_user(["test", "-r", str(path)])
        assert proc.returncode == 0, f"spaitra cannot read {path}"

    core = _read_env(runtime_core_env)
    ocr = _read_env(runtime_ocr_env)
    expected_core = {
        "ENABLE_DEPTH": "1",
        "ENABLE_OCR": "1",
        "SAVE_VRAM": "0",
        "STARTUP_WARM_MODE": "full",
        "ENABLE_CORE_SERVICE": "1",
    }
    for key, expected in expected_core.items():
        assert core.get(key) == expected, f"{key} expected {expected!r} got {core.get(key)!r}"
    assert ocr.get("OCR_MAX_CONCURRENCY") == "4", f"OCR_MAX_CONCURRENCY got {ocr.get('OCR_MAX_CONCURRENCY')!r}"
    assert ocr.get("OCR_USE_GPU") == "1", f"OCR_USE_GPU got {ocr.get('OCR_USE_GPU')!r}"
    assert ocr.get("OCR_GPU_REQUIRED") == "1", f"OCR_GPU_REQUIRED got {ocr.get('OCR_GPU_REQUIRED')!r}"


def test_runtime_repo_dirs() -> None:
    for path in [_WORKSPACE_ROOT / "backend-copy" / "data", _WORKSPACE_ROOT / "backend-copy" / "logs"]:
        proc = _run_as_service_user(["bash", "-lc", f"cd {path} && : > .runpod_test_probe && rm -f .runpod_test_probe"])
        assert proc.returncode == 0, f"spaitra cannot read/write {path}: {proc.stderr or proc.stdout}"

    db_path = _WORKSPACE_ROOT / "backend-copy" / "data" / "memory.db"
    proc = _run_as_service_user(["test", "-e", str(db_path)])
    if proc.returncode == 0:
        proc = _run_as_service_user(["test", "-r", str(db_path)])
        assert proc.returncode == 0, f"spaitra cannot read database file {db_path}"


def test_supervisor_and_ports() -> None:
    core_env = _read_env(_WORKSPACE_ROOT / ".env")
    proc = subprocess.run(
        ["supervisorctl", "-c", "/etc/supervisor/spaitra-supervisord.conf", "status"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    statuses = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            statuses[parts[0]] = parts[1]
    for name in ["spaitra-sshd", "spaitra-ocr", "spaitra-core", "spaitra-tunnel", "ollama"]:
        assert name in statuses, f"missing supervisor program {name}"
    expected_running = ["spaitra-ocr", "spaitra-core", "spaitra-sshd"]
    if core_env.get("ENABLE_OLLAMA_SERVICE", "1") == "1":
        expected_running.append("ollama")
    if core_env.get("ENABLE_SRVUS", "1") == "1":
        expected_running.append("spaitra-tunnel")
    for name in expected_running:
        assert "RUNNING" in statuses[name], f"{name} not running: {statuses[name]}"

    sock = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, check=True)
    listeners = sock.stdout
    for listener in ["127.0.0.1:5000", "127.0.0.1:8002"]:
        assert listener in listeners, f"missing listener {listener}"


def test_health_and_dependencies() -> None:
    core_env = _read_env(_WORKSPACE_ROOT / ".env")
    headers = {"X-API-Key": core_env.get("API_KEY", "")} if core_env.get("API_KEY") else {}
    code, _ = _http_json("http://127.0.0.1:5000/health")
    assert code == 200, f"/health status={code}"
    code, payload = _http_json("http://127.0.0.1:8002/health")
    assert code == 200, f"ocr /health status={code}"
    ocr_payload = (payload or {}).get("ocr", {})
    assert ocr_payload.get("gpu_requested") is True, f"gpu_requested={ocr_payload.get('gpu_requested')!r}"
    assert ocr_payload.get("gpu_required") is True, f"gpu_required={ocr_payload.get('gpu_required')!r}"
    assert ocr_payload.get("gpu_available") is True, f"gpu_available={ocr_payload.get('gpu_available')!r}"
    assert ocr_payload.get("gpu_state") == "ready", f"gpu_state={ocr_payload.get('gpu_state')!r}"
    code, payload = _http_json("http://127.0.0.1:5000/health/dependencies", headers=headers)
    assert code == 200, f"/health/dependencies status={code}"
    deps = (payload or {}).get("dependencies", {})
    assert deps.get("ffmpeg", {}).get("available"), "ffmpeg unavailable"
    ocr = deps.get("ocr", {})
    assert (not ocr.get("enabled")) or ocr.get("reachable"), "ocr unreachable"


if __name__ == "__main__":
    if not _WORKSPACE_ROOT.parent.exists() and not _STRICT:
        raise SystemExit(_skip("/workspace is missing; not running on the RunPod host."))

    for name, fn in [
        ("runpod_runtime:paths_and_user", test_paths_and_user),
        ("runpod_runtime:env_permissions_and_flags", test_env_permissions_and_flags),
        ("runpod_runtime:runtime_repo_dirs", test_runtime_repo_dirs),
        ("runpod_runtime:supervisor_and_ports", test_supervisor_and_ports),
        ("runpod_runtime:health_and_dependencies", test_health_and_dependencies),
    ]:
        _runner.run(name, fn)

    write_json(
        _RUN_DIR,
        "summary.json",
        {
            "suite": "runpod_runtime",
            "workspace_root": str(_WORKSPACE_ROOT),
            "opt_root": str(_OPT_ROOT),
        },
    )
    raise SystemExit(_runner.summary())
