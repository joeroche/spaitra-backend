"""HTTP OCR microservice backed by PaddleOCR."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import io
import logging
import os
import time
from functools import lru_cache

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image

_API_KEY = os.environ.get("API_KEY", "").strip()
_OCR_MAX_CONCURRENCY = max(1, int(os.environ.get("OCR_MAX_CONCURRENCY", "1")))
_OCR_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("OCR_REQUEST_TIMEOUT_SECONDS", "40"))
_OCR_SEMAPHORE_ACQUIRE_TIMEOUT_SECONDS = float(os.environ.get("OCR_SEMAPHORE_ACQUIRE_TIMEOUT_SECONDS", "1.0"))
_OCR_THROTTLE_RETRY_AFTER_SECONDS = max(1, int(os.environ.get("OCR_THROTTLE_RETRY_AFTER_SECONDS", "2")))
_OCR_RATE_LIMIT_PER_MIN = max(0, int(os.environ.get("OCR_RATE_LIMIT_PER_MIN", "120")))
_OCR_USE_GPU_REQUESTED = os.environ.get("OCR_USE_GPU", "1").strip().lower() in {"1", "true", "yes", "on"}
_OCR_GPU_REQUIRED = os.environ.get("OCR_GPU_REQUIRED", "0").strip().lower() in {"1", "true", "yes", "on"}
_OCR_GPU_DEVICE_ID = os.environ.get("OCR_GPU_DEVICE_ID", "0").strip() or "0"
_OCR_SEMAPHORE = asyncio.Semaphore(_OCR_MAX_CONCURRENCY)
_OCR_RATE_LOCK = asyncio.Lock()
_OCR_RATE_WINDOW_SECONDS = 60.0
_OCR_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_LOG = logging.getLogger("spaitra.ocr")


def _bool_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _gpu_runtime_state() -> tuple[bool, str]:
    try:
        import paddle
    except ImportError:
        return False, "paddle_not_installed"
    try:
        if not bool(paddle.device.is_compiled_with_cuda()):
            return False, "paddle_not_cuda_build"
        if int(paddle.device.cuda.device_count()) <= 0:
            return False, "no_cuda_devices"
    except RuntimeError:
        return False, "cuda_runtime_unavailable"
    return True, "ready"


def _decode_image(data: bytes) -> Image.Image | None:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data))
    except OSError:
        return None


def _resize_for_ocr(image: Image.Image) -> Image.Image:
    max_side = int(os.environ.get("OCR_MAX_SIDE", "2000"))
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image
    scale = max_side / float(largest_side)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _preferred_ocr_device(use_gpu: bool) -> str:
    if use_gpu:
        return f"gpu:{_OCR_GPU_DEVICE_ID}"
    return "cpu"


def _append_segment(
    segments: list[dict[str, float | str]],
    texts: list[str],
    confidences: list[float],
    *,
    text: str | None,
    confidence: float | int | None,
    min_conf: float,
) -> None:
    if text is None:
        return
    if confidence is None:
        confidence = 1.0
    score = float(confidence)
    if score < min_conf:
        return
    normalized_text = str(text).strip()
    if not normalized_text:
        return
    segments.append({"text": normalized_text, "confidence": score})
    texts.append(normalized_text)
    confidences.append(score)


def _collect_segments(
    raw: object,
    *,
    min_conf: float,
    segments: list[dict[str, float | str]],
    texts: list[str],
    confidences: list[float],
) -> None:
    if raw is None:
        return

    if hasattr(raw, "to_dict"):
        try:
            raw = raw.to_dict()
        except Exception:
            pass

    if isinstance(raw, dict):
        rec_texts = raw.get("rec_text")
        rec_scores = raw.get("rec_score")
        if isinstance(rec_texts, list):
            scores = rec_scores if isinstance(rec_scores, list) else []
            for idx, text in enumerate(rec_texts):
                score = scores[idx] if idx < len(scores) else 1.0
                _append_segment(
                    segments,
                    texts,
                    confidences,
                    text=text,
                    confidence=score,
                    min_conf=min_conf,
                )
            return

        _append_segment(
            segments,
            texts,
            confidences,
            text=raw.get("text"),
            confidence=raw.get("score", raw.get("confidence")),
            min_conf=min_conf,
        )
        return

    if isinstance(raw, (list, tuple)):
        if len(raw) >= 2 and isinstance(raw[1], (list, tuple)):
            pair = raw[1]
            if len(pair) >= 2:
                _append_segment(
                    segments,
                    texts,
                    confidences,
                    text=pair[0],
                    confidence=pair[1],
                    min_conf=min_conf,
                )
                return
        for item in raw:
            _collect_segments(
                item,
                min_conf=min_conf,
                segments=segments,
                texts=texts,
                confidences=confidences,
            )
        return

    _append_segment(
        segments,
        texts,
        confidences,
        text=str(raw),
        confidence=1.0,
        min_conf=min_conf,
    )


@lru_cache(maxsize=1)
def _get_ocr_engine():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR dependencies are not installed. "
            "Install the OCR environment with `pip install -e .[ocr]`."
        ) from exc

    lang = os.environ.get("OCR_LANG", "en")
    use_angle_cls = _bool_env("OCR_USE_ANGLE_CLS")
    enable_mkldnn = _bool_env("OCR_ENABLE_MKLDNN", "false")
    gpu_available, gpu_state = _gpu_runtime_state()
    use_gpu = _OCR_USE_GPU_REQUESTED and gpu_available
    if _OCR_USE_GPU_REQUESTED and not gpu_available:
        msg = f"OCR_USE_GPU requested but unavailable ({gpu_state})"
        if _OCR_GPU_REQUIRED:
            raise RuntimeError(msg)
        _LOG.warning("%s; using CPU OCR backend", msg)

    kwargs = dict(
        lang=lang,
        use_doc_orientation_classify=use_angle_cls,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=enable_mkldnn,
    )
    device = _preferred_ocr_device(use_gpu)
    kwargs["device"] = device
    try:
        return PaddleOCR(**kwargs)
    except (TypeError, ValueError) as exc:
        _LOG.info(
            "PaddleOCR device init unsupported (%s); retrying legacy use_gpu path",
            exc,
        )
        kwargs.pop("device", None)
        kwargs["use_gpu"] = bool(use_gpu)
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as legacy_exc:
            _LOG.info(
                "PaddleOCR legacy use_gpu init unsupported (%s); retrying without explicit device",
                legacy_exc,
            )
            kwargs.pop("use_gpu", None)
            return PaddleOCR(**kwargs)


def _extract_text(image: Image.Image) -> tuple[dict, float]:
    """Extract text from image. Returns (result_dict, elapsed_ms)."""
    start_time = time.time()
    engine = _get_ocr_engine()
    rgb = np.array(image.convert("RGB"))
    try:
        result = engine.ocr(rgb, cls=False)
    except TypeError:
        # PaddleOCR v3 removed the cls kwarg on predict/ocr path.
        result = engine.ocr(rgb)
    min_conf = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.3"))

    segments: list[dict[str, float | str]] = []
    texts: list[str] = []
    confidences: list[float] = []
    payload = result[0] if isinstance(result, list) and result else result
    _collect_segments(
        payload,
        min_conf=min_conf,
        segments=segments,
        texts=texts,
        confidences=confidences,
    )

    elapsed_ms = (time.time() - start_time) * 1000
    result_dict = {
        "text": " ".join(texts).strip(),
        "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "segments": segments,
    }
    return result_dict, elapsed_ms


def _run_ocr_sync(image: Image.Image) -> dict:
    resized_image = _resize_for_ocr(image)
    result, ocr_time_ms = _extract_text(resized_image)
    result["_ocr_time_ms"] = ocr_time_ms
    return result


app = FastAPI(title="Spaitra OCR Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    gpu_available, gpu_state = _gpu_runtime_state()
    return {
        "status": "ok",
        "ocr": {
            "gpu_requested": _OCR_USE_GPU_REQUESTED,
            "gpu_required": _OCR_GPU_REQUIRED,
            "gpu_available": gpu_available,
            "gpu_state": gpu_state,
            "gpu_device_id": _OCR_GPU_DEVICE_ID,
            "max_concurrency": _OCR_MAX_CONCURRENCY,
        },
    }


@app.post("/ocr")
async def run_ocr(
    request: Request,
    image: UploadFile = File(...),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    if _API_KEY and api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    if not image.filename:
        raise HTTPException(status_code=400, detail="missing image")

    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not client_ip and request.client is not None:
        client_ip = request.client.host or "unknown"
    if not client_ip:
        client_ip = "unknown"

    if _OCR_RATE_LIMIT_PER_MIN > 0:
        now = time.monotonic()
        async with _OCR_RATE_LOCK:
            bucket = _OCR_RATE_BUCKETS[client_ip]
            cutoff = now - _OCR_RATE_WINDOW_SECONDS
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= _OCR_RATE_LIMIT_PER_MIN:
                retry_after = max(1, int(bucket[0] + _OCR_RATE_WINDOW_SECONDS - now))
                _LOG.warning(
                    "ocr_rate_limited client_ip=%s limit_per_min=%s retry_after_s=%s",
                    client_ip,
                    _OCR_RATE_LIMIT_PER_MIN,
                    retry_after,
                )
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={
                        "error": "rate_limited",
                        "message": "OCR request rate is limited. Retry later.",
                        "retry_after_seconds": retry_after,
                        "limit_per_minute": _OCR_RATE_LIMIT_PER_MIN,
                    },
                )
            bucket.append(now)

    acquired = False
    try:
        try:
            await asyncio.wait_for(
                _OCR_SEMAPHORE.acquire(),
                timeout=_OCR_SEMAPHORE_ACQUIRE_TIMEOUT_SECONDS,
            )
            acquired = True
        except TimeoutError:
            _LOG.warning(
                "ocr_throttled capacity=%s retry_after_s=%s",
                _OCR_MAX_CONCURRENCY,
                _OCR_THROTTLE_RETRY_AFTER_SECONDS,
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(_OCR_THROTTLE_RETRY_AFTER_SECONDS)},
                content={
                    "error": "server_busy",
                    "message": "OCR capacity is full. Retry shortly.",
                    "retry_after_seconds": _OCR_THROTTLE_RETRY_AFTER_SECONDS,
                    "capacity": _OCR_MAX_CONCURRENCY,
                },
            )

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty image")

        decode_t0 = time.time()
        try:
            pil_image = _decode_image(data)
        except OSError as exc:
            raise HTTPException(status_code=400, detail="invalid image") from exc
        if pil_image is None:
            raise HTTPException(status_code=400, detail="invalid image")
        decode_ms = (time.time() - decode_t0) * 1000

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_ocr_sync, pil_image),
                timeout=_OCR_REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            _LOG.warning(
                "ocr_timeout timeout_s=%s capacity=%s",
                _OCR_REQUEST_TIMEOUT_SECONDS,
                _OCR_MAX_CONCURRENCY,
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "ocr_timeout",
                    "message": "OCR request timed out under current load.",
                    "timeout_seconds": _OCR_REQUEST_TIMEOUT_SECONDS,
                },
            )
        total_ms = decode_ms + float(result.get("_ocr_time_ms", 0.0))
        result["_decode_ms"] = decode_ms
        result["_total_ms"] = total_ms
        return result
    except (RuntimeError, NotImplementedError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if acquired:
            _OCR_SEMAPHORE.release()
