"""Central state policy for voice gating, ASR context, and LLM intent bias."""
from __future__ import annotations


def normalize_mode(mode: object) -> str:
    value = str(mode or "idle").strip().lower()
    return value or "idle"


def _normalize_pending_action(context: dict) -> str:
    pending = str(context.get("pending_action") or "").strip().lower()
    if pending in {"scan", "remember", "describe_scene", "describe_target"}:
        return pending
    return ""


def _normalize_pending_voice_action(context: dict) -> str:
    pending = str(context.get("pending_voice_action") or "").strip().lower()
    if pending in {"remember_label", "find_target"}:
        return pending
    return ""


def resolve_voice_policy(mode: object, context: object) -> dict:
    mode_value = normalize_mode(mode)
    ctx = context if isinstance(context, dict) else {}
    pending_action = _normalize_pending_action(ctx)
    pending_voice_action = _normalize_pending_voice_action(ctx)

    policy_id = "idle_home"
    whisper_boost = [
        "scan", "teach", "remember", "find", "where",
        "describe", "settings", "read",
    ]
    allowed_global_intents = ["scan", "remember", "find", "ask", "open_settings", "navigate_back"]
    guidance: dict[str, str] = {}
    include_known_labels = True
    include_recent_rooms = True
    include_common_query_terms = True

    if pending_voice_action == "remember_label":
        policy_id = "clarify_teach_label"
        whisper_boost = ["teach", "remember", "save", "item", "name", "called", "my"]
        allowed_global_intents = ["remember", "open_settings", "navigate_back"]
        include_known_labels = False
        include_recent_rooms = False
        include_common_query_terms = False
    elif pending_voice_action == "find_target":
        policy_id = "clarify_find_target"
        whisper_boost = ["find", "where", "item", "thing", "wallet", "keys", "receipt"]
        allowed_global_intents = ["find", "open_settings", "navigate_back"]
        include_recent_rooms = False

    if pending_voice_action == "":
        if mode_value == "focused_on_item":
            policy_id = "focused_item_scan_browse"
            whisper_boost = [
                "describe", "read", "text", "says", "export", "copy",
                "rename", "wrong", "correct", "right", "where is this",
            ]
            allowed_global_intents = ["scan", "find", "ask", "open_settings", "navigate_back"]
            guidance = {
                "remember": "Return home to teach a new item. Say go back, then teach.",
            }
            include_recent_rooms = False
            include_common_query_terms = False
        elif mode_value == "awaiting_location":
            policy_id = "awaiting_location_capture"
            whisper_boost = [
                "kitchen", "bedroom", "bathroom", "living room",
                "office", "garage", "hallway",
            ]
            allowed_global_intents = []
            include_known_labels = False
            include_common_query_terms = False
        elif mode_value == "awaiting_confirmation":
            policy_id = "awaiting_confirmation_binary"
            whisper_boost = ["yes", "no", "confirm", "cancel", "correct", "wrong"]
            allowed_global_intents = []
            include_known_labels = False
            include_recent_rooms = False
            include_common_query_terms = False
        elif mode_value == "onboarding_teach":
            policy_id = "onboarding_teach_step"
            whisper_boost = ["teach", "remember", "save", "this is", "my"]
            allowed_global_intents = ["remember", "open_settings", "navigate_back"]
            include_known_labels = False
            include_recent_rooms = False
            include_common_query_terms = False
        elif mode_value == "onboarding_await_scan":
            policy_id = "onboarding_scan_step"
            whisper_boost = ["scan", "camera", "hold", "phone"]
            allowed_global_intents = ["scan", "open_settings", "navigate_back"]
            include_known_labels = False
            include_recent_rooms = False
            include_common_query_terms = False
        elif mode_value == "awaiting_image":
            include_known_labels = False
            include_recent_rooms = False
            include_common_query_terms = False
            if pending_action == "scan":
                policy_id = "awaiting_image_scan"
                whisper_boost = ["scan", "camera", "hold", "phone"]
            elif pending_action == "remember":
                policy_id = "awaiting_image_remember"
                whisper_boost = ["teach", "remember", "item", "camera", "hold", "phone"]
            elif pending_action in {"describe_scene", "describe_target"}:
                policy_id = "awaiting_image_describe"
                whisper_boost = ["describe", "what i am looking at", "describe this", "camera", "hold", "phone"]
            else:
                policy_id = "awaiting_image_generic"
                whisper_boost = ["camera", "hold", "phone", "image"]
            allowed_global_intents = []

    return {
        "policy_id": policy_id,
        "mode": mode_value,
        "pending_action": pending_action,
        "pending_voice_action": pending_voice_action,
        "allowed_global_intents": allowed_global_intents,
        "guidance": guidance,
        "whisper_boost_terms": whisper_boost,
        "include_known_labels": include_known_labels,
        "include_recent_rooms": include_recent_rooms,
        "include_common_query_terms": include_common_query_terms,
    }
