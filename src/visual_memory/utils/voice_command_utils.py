"""Shared helpers for state-aware voice command parsing."""
from __future__ import annotations

import re

_TEACH_PATTERNS = [
    re.compile(r"\b(?:remember|teach|this is)\b.*?\b(?:as|called|named)\b\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:this is my|that's my|it's my)\b\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:save this as|store this as|call this)\b\s+(.+)", re.IGNORECASE),
]

_LEADING_ARTICLES_RE = re.compile(r"^(?:my|a|an|the)\s+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

_ROOM_WORDS = {
    "bathroom",
    "bedroom",
    "garage",
    "hallway",
    "kitchen",
    "living room",
    "office",
}

_RESERVED_LABELS = {
    "ask",
    "back",
    "cancel",
    "describe",
    "export",
    "find",
    "home",
    "read",
    "remember",
    "rename",
    "save",
    "scan",
    "settings",
    "stop",
    "teach",
}

_VAGUE_FIND_TARGETS = {
    "",
    "it",
    "item",
    "object",
    "one",
    "something",
    "that",
    "thing",
    "this",
}

_AFFIRMATIVE_WORDS = {"yes", "yeah", "yep", "confirm", "correct", "do it", "sure"}
_NEGATIVE_WORDS = {"no", "nope", "cancel", "stop", "never mind"}


def extract_teach_label(transcription: str) -> str | None:
    text = (transcription or "").strip()
    if not text:
        return None
    for pattern in _TEACH_PATTERNS:
        match = pattern.search(text)
        if match:
            label = match.group(1).strip().strip("\"'").strip(".,!?;:")
            if label:
                return label
    return None


def normalize_teach_reply(text: str) -> str:
    cleaned = re.sub(
        r"^(teach|remember|save|learn|this is|it's|it is)\s+(me\s+)?",
        "",
        (text or "").strip(),
        flags=re.IGNORECASE,
    )
    cleaned = _LEADING_ARTICLES_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip("\"'").strip(".,!?;:")
    cleaned = _WS_RE.sub(" ", cleaned)
    return cleaned.lower()


def needs_label_clarification(label: str) -> bool:
    normalized = normalize_teach_reply(label)
    if not normalized:
        return True
    if len(normalized) <= 1:
        return True
    if normalized in _RESERVED_LABELS:
        return True
    if normalized in _ROOM_WORDS:
        return True
    return False


def needs_find_target_clarification(query: str) -> bool:
    normalized = _WS_RE.sub(" ", (query or "").strip().lower()).strip(" ?.!,'\"")
    return normalized in _VAGUE_FIND_TARGETS


def resolve_confirmation(command_text: str) -> bool | None:
    q = _WS_RE.sub(" ", (command_text or "").strip().lower())
    if not q:
        return None
    confirmed = any(token in q for token in _AFFIRMATIVE_WORDS)
    denied = any(token in q for token in _NEGATIVE_WORDS)
    if confirmed and not denied:
        return True
    if denied and not confirmed:
        return False
    return None


def resolve_pending_voice_action(command_text: str, context: dict | None) -> dict | None:
    if not isinstance(context, dict):
        return None
    pending = str(context.get("pending_voice_action") or "").strip().lower()
    if pending not in {"remember_label", "find_target"}:
        return None

    lowered = (command_text or "").strip().lower()
    if not lowered:
        return {"command": "clarify_label" if pending == "remember_label" else "clarify_find"}
    if lowered in {"back", "go back", "home", "cancel"}:
        return {"command": "navigate_back"}
    if lowered in {"settings", "open settings", "preferences"}:
        return {"command": "open_settings"}

    if pending == "remember_label":
        label = normalize_teach_reply(command_text)
        if needs_label_clarification(label):
            return {"command": "clarify_label"}
        return {"command": "remember", "label": label}

    target = _WS_RE.sub(" ", (command_text or "").strip()).strip()
    if needs_find_target_clarification(target):
        return {"command": "clarify_find"}
    return {"command": "find", "query": target}
