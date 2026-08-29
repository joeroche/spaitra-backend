"""Image embedding module using DINOv3."""
from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from visual_memory.config import Settings
from visual_memory.utils import get_logger
from visual_memory.utils.device_utils import get_device

_defaults = Settings()
_log = get_logger(__name__)


class ImageEmbedder:
    """
    Image embedder using DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m).

    Returns 1024-dim L2-normalized embeddings from the pooler output.
    Self-supervised vision-only model; better object-level discrimination
    than CLIP for same-scene objects.

    Requires transformers >= 4.56.0 for DINOv3 support.
    """

    def __init__(
        self,
        pretrained_model_name: str = _defaults.image_embedder_model,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device if device else get_device())
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name).to(self.device).eval()

    def to_cpu(self) -> None:
        """Move model weights to CPU RAM."""
        if self.device != torch.device("cpu"):
            self.model.to("cpu")
            self.device = torch.device("cpu")

    def to_gpu(self) -> None:
        """Restore model weights to GPU."""
        target = torch.device(get_device())
        if self.device != target:
            self.model.to(target)
            self.device = target

    def _prepare_inputs(self, image: Image.Image):
        """Apply the canonical DINOv3 processor and move tensors to the model device."""
        return self.processor(images=image, return_tensors="pt").to(self.device)

    def _forward(self, image: Image.Image):
        """Run a single DINOv3 forward pass under inference mode."""
        inputs = self._prepare_inputs(image)
        with torch.inference_mode():
            return self.model(**inputs)

    def embed(self, image: Image.Image) -> torch.Tensor:
        """Embed a single image. Returns (1, 1024) L2-normalized tensor."""
        outputs = self._forward(image)
        return F.normalize(outputs.pooler_output.detach().cpu(), dim=1)

    def extract_patch_tokens(
        self,
        image: Image.Image,
        include_cls_token: bool = False,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """Return DINOv3 patch tokens for verifier-style local matching.

        Args:
            image:
                PIL image processed through the same AutoImageProcessor path as embed().
            include_cls_token:
                When True, keep the leading CLS token. Default False returns only patch tokens.
            return_cpu:
                When True, detach and move tokens to CPU. When False, keep them on self.device.

        Returns:
            Tensor shaped (batch, num_tokens, hidden_dim). For DINOv3 ViT-L this is
            typically (1, num_patches, 1024) when include_cls_token=False.

        Raises:
            RuntimeError:
                If the model output does not expose last_hidden_state.
        """
        mem_before = None
        if self.device.type == "cuda":
            mem_before = torch.cuda.memory_allocated(self.device)

        outputs = self._forward(image)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None:
            if hasattr(outputs, "items"):
                available = sorted(k for k, v in outputs.items() if v is not None)
            else:
                available = sorted(k for k, v in vars(outputs).items() if v is not None)
            raise RuntimeError(
                "DINOv3 outputs do not expose last_hidden_state; "
                f"available outputs: {available}"
            )
        if not include_cls_token and tokens.shape[1] > 0:
            tokens = tokens[:, 1:, :]

        detached = tokens.detach()
        result = detached.cpu() if return_cpu else detached

        mem_after = None
        if self.device.type == "cuda":
            mem_after = torch.cuda.memory_allocated(self.device)
        _log.debug({
            "event": "dinov3_patch_tokens_extracted",
            "device": str(self.device),
            "shape": list(result.shape),
            "include_cls_token": include_cls_token,
            "return_cpu": return_cpu,
            "cuda_mem_before": mem_before,
            "cuda_mem_after": mem_after,
        })
        return result

    def embed_patch_tokens(
        self,
        image: Image.Image,
        include_cls_token: bool = False,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """Backward-compatible alias for verifier probes expecting embed_patch_tokens()."""
        return self.extract_patch_tokens(
            image,
            include_cls_token=include_cls_token,
            return_cpu=return_cpu,
        )

    def batch_embed(self, images: List[Image.Image]) -> torch.Tensor:
        """Embed a batch of images in a single forward pass.

        Returns (N, 1024) L2-normalized tensor. Preferred over calling
        embed() in a loop on CUDA/MPS - processes all images at once.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return F.normalize(outputs.pooler_output.detach().cpu(), dim=1)
