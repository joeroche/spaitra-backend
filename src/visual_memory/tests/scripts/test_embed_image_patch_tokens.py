"""Unit tests for ImageEmbedder patch-token extraction without model loading."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from PIL import Image

from visual_memory.engine.embedding.embed_image import ImageEmbedder
from visual_memory.tests.scripts.test_harness import TestRunner

_runner = TestRunner("embed_image_patch_tokens")


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None


def _make_embedder(fake_hidden: torch.Tensor) -> ImageEmbedder:
    embedder = ImageEmbedder.__new__(ImageEmbedder)
    embedder.device = torch.device("cpu")
    embedder._forward = lambda image: SimpleNamespace(last_hidden_state=fake_hidden)
    return embedder


def test_extract_patch_tokens_drops_cls_by_default():
    fake_hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    embedder = _make_embedder(fake_hidden)

    patch_tokens = embedder.extract_patch_tokens(Image.new("RGB", (8, 8)))

    assert tuple(patch_tokens.shape) == (1, 5, 4), f"unexpected shape: {tuple(patch_tokens.shape)}"
    assert torch.equal(patch_tokens, fake_hidden[:, 1:, :]), "expected CLS token to be removed"


def test_extract_patch_tokens_can_keep_cls_on_cpu():
    fake_hidden = torch.randn(1, 7, 4)
    embedder = _make_embedder(fake_hidden)

    patch_tokens = embedder.extract_patch_tokens(
        Image.new("RGB", (8, 8)),
        include_cls_token=True,
        return_cpu=True,
    )

    assert tuple(patch_tokens.shape) == (1, 7, 4), f"unexpected shape: {tuple(patch_tokens.shape)}"
    assert patch_tokens.device.type == "cpu"


def test_extract_patch_tokens_raises_when_hidden_state_missing():
    embedder = ImageEmbedder.__new__(ImageEmbedder)
    embedder.device = torch.device("cpu")
    embedder._forward = lambda image: SimpleNamespace(pooler_output=torch.randn(1, 1024))

    try:
        embedder.extract_patch_tokens(Image.new("RGB", (8, 8)))
        raise AssertionError("expected RuntimeError when last_hidden_state is missing")
    except RuntimeError as exc:
        assert "last_hidden_state" in str(exc)
        assert "pooler_output" in str(exc)


import visual_memory.engine.embedding.embed_image as _embed_image_mod

_old_log = _embed_image_mod._log
_embed_image_mod._log = _DummyLogger()

for name, fn in [
    ("embed_image:drop_cls", test_extract_patch_tokens_drops_cls_by_default),
    ("embed_image:keep_cls", test_extract_patch_tokens_can_keep_cls_on_cpu),
    ("embed_image:missing_hidden_state", test_extract_patch_tokens_raises_when_hidden_state_missing),
]:
    _runner.run(name, fn)

_embed_image_mod._log = _old_log
sys.exit(_runner.summary())
