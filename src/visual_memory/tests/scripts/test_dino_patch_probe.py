"""DINOv3 patch-token probe for Step B.0.

Usage:
    python -m visual_memory.tests.scripts.test_dino_patch_probe
"""
from __future__ import annotations

import sys

from PIL import Image

from visual_memory.engine.embedding import ImageEmbedder
from visual_memory.tests.scripts.test_harness import TestRunner

_runner = TestRunner("dino_patch_probe")


def test_extract_patch_tokens_api():
    """Load DINOv3 once, verify last_hidden_state access, and print shapes."""
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    embedder = ImageEmbedder()

    outputs = embedder._forward(image)
    hidden = getattr(outputs, "last_hidden_state", None)
    assert hidden is not None, "expected DINOv3 outputs.last_hidden_state to be available"
    assert len(hidden.shape) == 3, f"expected 3D hidden state, got {tuple(hidden.shape)}"

    patch_tokens = embedder.extract_patch_tokens(image, include_cls_token=False, return_cpu=True)
    with_cls = embedder.extract_patch_tokens(image, include_cls_token=True, return_cpu=True)

    assert patch_tokens.shape[0] == 1, f"expected batch size 1, got {tuple(patch_tokens.shape)}"
    assert patch_tokens.shape[-1] == hidden.shape[-1], (
        f"patch dim mismatch: patch={patch_tokens.shape[-1]} hidden={hidden.shape[-1]}"
    )
    assert with_cls.shape[1] == patch_tokens.shape[1] + 1, (
        f"expected CLS-inclusive tokens to add one token: no_cls={tuple(patch_tokens.shape)} "
        f"with_cls={tuple(with_cls.shape)}"
    )

    print(f"outputs.last_hidden_state shape: {tuple(hidden.shape)}")
    print(f"extract_patch_tokens(include_cls_token=False) shape: {tuple(patch_tokens.shape)}")
    print(f"extract_patch_tokens(include_cls_token=True) shape: {tuple(with_cls.shape)}")


_runner.run("dino:patch_tokens", test_extract_patch_tokens_api)
sys.exit(_runner.summary())
