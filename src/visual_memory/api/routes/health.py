import time
import urllib.error
import urllib.parse
import urllib.request

from flask import Blueprint, jsonify

from visual_memory.api.pipelines import get_settings
from visual_memory.utils.audio_utils import ensure_ffmpeg_available

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    """Return a lightweight liveness response for probes and quick checks."""
    return jsonify({"status": "ok"})


def _ocr_health() -> dict:
    settings = get_settings()
    if not settings.enable_ocr:
        return {
            "enabled": False,
            "reachable": None,
            "latency_ms": None,
            "url": settings.ocr_health_url or settings.ocr_service_url,
        }

    url = settings.ocr_health_url
    if not url:
        parsed = urllib.parse.urlparse(settings.ocr_service_url)
        url = f"{parsed.scheme}://{parsed.netloc}/health"

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            reachable = resp.status == 200
    except (urllib.error.URLError, OSError):
        reachable = False
    latency_ms = round((time.monotonic() - t0) * 1000, 1)

    return {
        "enabled": True,
        "reachable": reachable,
        "latency_ms": latency_ms if reachable else None,
        "url": url,
    }


@health_bp.get("/health/dependencies")
def health_dependencies():
    """Return readiness of external tools/services required by production routes."""
    ffmpeg_ok = True
    ffmpeg_error = None
    try:
        ensure_ffmpeg_available()
    except RuntimeError as exc:
        ffmpeg_ok = False
        ffmpeg_error = str(exc)

    ocr = _ocr_health()
    ocr_ok = (ocr["enabled"] is False) or (ocr["reachable"] is True)
    status = "ok" if (ffmpeg_ok and ocr_ok) else "degraded"

    return jsonify({
        "status": status,
        "dependencies": {
            "ffmpeg": {
                "available": ffmpeg_ok,
                "error": ffmpeg_error,
            },
            "ocr": ocr,
        },
    })
