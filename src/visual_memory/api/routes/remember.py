import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request

from visual_memory.api.pipelines import (
    get_database,
    get_remember_pipeline,
    get_user_settings,
    reload_scan_database_if_loaded,
)
from visual_memory.engine.vlm import get_vlm_pipeline
from visual_memory.utils import get_logger


remember_bp = Blueprint("remember", __name__)
_log = get_logger(__name__)
_UPLOADS_DIR = Path(__file__).resolve().parents[4] / "uploads"
_EXEMPLARS_DIR = Path(__file__).resolve().parents[4] / "data" / "exemplars"
_VLM_STORE_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _save_upload(image_file) -> Path:
    suffix = Path(image_file.filename).suffix if image_file.filename else ".jpg"
    _UPLOADS_DIR.mkdir(exist_ok=True)
    path = _UPLOADS_DIR / f"remember_{int(time.time() * 1000)}_{uuid4().hex}{suffix}"
    image_file.save(str(path))
    return path


def _persist_exemplar(tmp_path: Path, label: str) -> Path:
    """Move tmp_path to a permanent per-label exemplars directory and return new path."""
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:64]
    dest_dir = _EXEMPLARS_DIR / safe_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / tmp_path.name
    shutil.move(str(tmp_path), str(dest))
    return dest


def _extract_item_id(result: dict) -> int | None:
    data = result.get("result")
    if not isinstance(data, dict):
        return None
    raw_item_id = data.get("item_id")
    try:
        item_id = int(raw_item_id)
    except (TypeError, ValueError):
        return None
    if item_id <= 0:
        return None
    return item_id


def _store_vlm_description(item_id: int | None, label: str, image_path: Path) -> None:
    try:
        perf_cfg = get_user_settings().get_performance_config()
        if not perf_cfg.vlm_enabled:
            return
        description = get_vlm_pipeline().describe(
            image_path,
            timeout=float(perf_cfg.vlm_timeout_seconds),
        )
        if not description:
            return
        if item_id is None:
            _log.warning({
                "event": "remember_vlm_description_skipped_missing_item_id",
                "label": label,
            })
            return
        get_database().update_item_vlm_description_by_id(item_id, description)
        _log.info({
            "event": "remember_vlm_description_stored",
            "label": label,
            "item_id": item_id,
            "description_len": len(description),
        })
    except Exception as exc:
        _log.warning({
            "event": "remember_vlm_description_failed",
            "label": label,
            "item_id": item_id,
            "error": str(exc),
        })


def _after_exemplar_persisted(item_id: int | None, label: str, path: Path) -> None:
    try:
        if item_id is None:
            _log.warning({
                "event": "remember_image_path_update_skipped_missing_item_id",
                "label": label,
            })
            return
        get_database().update_item_image_path_by_id(item_id, str(path))
    except Exception as exc:
        _log.warning({
            "event": "remember_image_path_update_failed",
            "label": label,
            "item_id": item_id,
            "error": str(exc),
        })
    _VLM_STORE_EXECUTOR.submit(_store_vlm_description, item_id, label, path)


def _remember_single(image_file, prompt: str) -> tuple[dict, int]:
    tmp_path = _save_upload(image_file)
    try:
        result = get_remember_pipeline().run(tmp_path, prompt)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return {"error": str(exc)}, 500

    if result.get("success"):
        try:
            item_id = _extract_item_id(result)
            exemplar_path = _persist_exemplar(tmp_path, prompt)
            _after_exemplar_persisted(item_id, prompt, exemplar_path)
        except Exception as exc:
            _log.warning({"event": "remember_exemplar_persist_failed", "error": str(exc)})
            tmp_path.unlink(missing_ok=True)
        try:
            reload_scan_database_if_loaded()
        except Exception as exc:
            _log.warning({
                "event": "remember_cache_reload_failed",
                "error": str(exc),
            })
    else:
        tmp_path.unlink(missing_ok=True)

    return result, 200


def _remember_multi(image_files, prompt: str) -> tuple[dict, int]:
    pipeline = get_remember_pipeline()
    tmp_paths = []
    result = {"success": False}
    best_idx = -1
    detected_count = 0
    try:
        for f in image_files:
            tmp_paths.append(_save_upload(f))

        try:
            scores = pipeline.detect_score_batch(tmp_paths, prompt)
        except Exception as exc:
            _log.warning({
                "event": "remember_detect_score_batch_failed",
                "error": str(exc),
                "images_tried": len(tmp_paths),
            })
            scores = []
            for p in tmp_paths:
                try:
                    score = pipeline.detect_score(p, prompt)
                except Exception as score_exc:
                    _log.warning({
                        "event": "remember_detect_score_single_failed",
                        "error": str(score_exc),
                        "image_path": str(p),
                    })
                    score = {
                        "detected": False,
                        "score": 0.0,
                        "blur_score": 0.0,
                        "is_dark": False,
                        "darkness_level": 0.0,
                        "second_pass_prompt": None,
                    }
                scores.append(score)

        detected_count = sum(1 for s in scores if s["detected"])
        best_idx = max(range(len(scores)), key=lambda i: scores[i]["score"]) if scores else -1

        if best_idx < 0 or not scores[best_idx]["detected"]:
            return {
                "success": False,
                "message": "No object detected in any of the provided images.",
                "images_tried": len(tmp_paths),
                "images_with_detection": 0,
                "result": None,
            }, 200

        try:
            result = pipeline.run(tmp_paths[best_idx], prompt)
        except Exception as exc:
            return {"error": str(exc)}, 500

    finally:
        for i, p in enumerate(tmp_paths):
            if i == best_idx and result.get("success"):
                try:
                    item_id = _extract_item_id(result)
                    exemplar_path = _persist_exemplar(p, prompt)
                    _after_exemplar_persisted(item_id, prompt, exemplar_path)
                except Exception as exc:
                    _log.warning({"event": "remember_exemplar_persist_failed", "error": str(exc)})
                    p.unlink(missing_ok=True)
            else:
                p.unlink(missing_ok=True)

    if result.get("success"):
        try:
            reload_scan_database_if_loaded()
        except Exception as exc:
            _log.warning({
                "event": "remember_multi_cache_reload_failed",
                "error": str(exc),
            })
        result["images_tried"] = len(tmp_paths)
        result["images_with_detection"] = detected_count

    return result, 200


def process_remember_request(prompt: str, image_file=None, image_files=None) -> tuple[dict, int]:
    normalized_prompt = (prompt or "").strip()
    if not normalized_prompt:
        return {"error": "missing field: prompt"}, 400
    if image_files:
        return _remember_multi(image_files, normalized_prompt)
    if image_file is None:
        return {"error": "missing field: image or images[]"}, 400
    return _remember_single(image_file, normalized_prompt)


@remember_bp.post("/remember")
def remember():
    result, status = process_remember_request(
        prompt=request.form.get("prompt", ""),
        image_file=request.files.get("image"),
        image_files=request.files.getlist("images[]"),
    )
    return jsonify(result), status
