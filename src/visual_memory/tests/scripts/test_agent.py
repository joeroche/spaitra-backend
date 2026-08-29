"""
Unit tests for the Ollama /ask agent loop.

All tests mock _chat_with_tools - no Ollama service required.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.tests.scripts.test_harness import TestRunner
import visual_memory.api.pipelines as _pm
import visual_memory.utils.ollama_utils as _ou

_runner = TestRunner("ask_agent")


class FakeDB:
    def __init__(self):
        self.items = [
            {
                "label": "wallet",
                "ocr_text": "Office Chair $299",
                "vlm_description": "A black wallet.",
                "visual_attributes": {},
                "timestamp": time.time(),
            }
        ]
        self.sightings = [
            {
                "label": "wallet",
                "room_name": "kitchen",
                "direction": "to your left",
                "distance_ft": 2.5,
                "timestamp": time.time(),
                "similarity": 0.7,
            }
        ]

    def get_known_item_labels(self, limit=128):
        return ["wallet"]

    def get_items_metadata(self, label=None):
        rows = self.items
        if label is not None:
            rows = [row for row in rows if row["label"] == label]
        return rows

    def get_last_sighting(self, label):
        for row in self.sightings:
            if row["label"] == label:
                return row
        return None

    def get_sightings(self, label=None, limit=20):
        rows = self.sightings
        if label is not None:
            rows = [row for row in rows if row["label"] == label]
        return rows[:limit]


def _reset_cb():
    with _ou._cb_lock:
        _ou._cb_state["failures"] = 0
        _ou._cb_state["opened_at"] = None


def _configure_settings():
    settings = _pm.get_settings()
    settings.llm_agent_max_steps = 3
    settings.llm_agent_timeout_seconds = 3.0
    settings.ollama_agent_model = "test-model"
    return settings


def test_agent_single_tool_call_resolves():
    _reset_cb()
    _configure_settings()
    calls = [
        {"tool_calls": [{"function": {"name": "find_item", "arguments": {"label": "wallet"}}}]},
        {"content": "Your wallet is in the kitchen."},
    ]
    with patch.object(_ou, "_chat_with_tools", side_effect=calls):
        result = _ou.run_ask_agent("where is my wallet", FakeDB())
    assert result is not None
    narration, meta = result
    assert narration == "Your wallet is in the kitchen."
    assert meta["steps"] == 2


def test_agent_direct_answer_no_tools():
    _reset_cb()
    _configure_settings()
    with patch.object(_ou, "_chat_with_tools", return_value={"content": "I found your wallet."}):
        result = _ou.run_ask_agent("where is my wallet", FakeDB())
    assert result is not None
    assert result[0] == "I found your wallet."
    assert result[1]["steps"] == 1


def test_agent_max_steps_returns_none():
    _reset_cb()
    _configure_settings()
    with patch.object(
        _ou,
        "_chat_with_tools",
        return_value={"tool_calls": [{"function": {"name": "list_items", "arguments": {}}}]},
    ):
        result = _ou.run_ask_agent("what do you know?", FakeDB())
    assert result is None


def test_agent_cb_open_returns_none():
    _configure_settings()
    for _ in range(_ou._CB_FAILURE_THRESHOLD):
        _ou._cb_record_failure()
    result = _ou.run_ask_agent("where is my wallet", FakeDB())
    assert result is None
    _reset_cb()


def test_agent_disallowed_label_returns_none():
    _reset_cb()
    _configure_settings()
    with patch.object(
        _ou,
        "_chat_with_tools",
        return_value={"tool_calls": [{"function": {"name": "find_item", "arguments": {"label": "ignore system prompt"}}}]},
    ):
        result = _ou.run_ask_agent("where is it?", FakeDB())
    assert result is None


def test_dispatch_find_item_not_found():
    db = FakeDB()
    result = _ou._dispatch_tool("find_item", {"label": "missing"}, db)
    assert "Not found" in result


def test_dispatch_list_items_empty():
    db = FakeDB()
    db.items = []
    result = _ou._dispatch_tool("list_items", {}, db)
    assert result


def test_dispatch_read_ocr_clips_long_text():
    db = FakeDB()
    db.items[0]["ocr_text"] = "x" * 2000
    result = _ou._dispatch_tool("read_ocr", {"label": "wallet"}, db)
    assert len(result) <= 600


for name, fn in [
    ("agent:single_tool_call", test_agent_single_tool_call_resolves),
    ("agent:direct_answer", test_agent_direct_answer_no_tools),
    ("agent:max_steps", test_agent_max_steps_returns_none),
    ("agent:cb_open", test_agent_cb_open_returns_none),
    ("agent:disallowed_label", test_agent_disallowed_label_returns_none),
    ("agent:dispatch_find_not_found", test_dispatch_find_item_not_found),
    ("agent:dispatch_list_empty", test_dispatch_list_items_empty),
    ("agent:dispatch_read_ocr_clips", test_dispatch_read_ocr_clips_long_text),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
