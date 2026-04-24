"""Frame embedding extractor with a small, modular backbone registry.

The default backbone is ``resnet18`` — it is the only hard dependency and
matches the original baseline (ImageNet-pretrained, 512-D pooled
features, CPU-friendly). The registry also exposes hooks for optional
backbones (``clip``, ``dino``) that are imported lazily; if the
underlying package is missing, :class:`EmbeddingExtractor` falls back
to ResNet-18 and logs a warning so the caller knows what happened.

Design notes
------------
- Backbone objects are frozen (eval + inference_mode). We deliberately do
  not train anything here.
- When ``l2_normalize=True`` (default) outputs are unit-norm, so the
  reference scorer can use cosine or Euclidean distance interchangeably.
- Preprocessing uses ImageNet stats for ResNet-18 / torchvision DINO
  checkpoints and CLIP's own constants for CLIP.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# Default ImageNet normalization (ResNet-18, most torchvision backbones).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CLIP uses its own constants.
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ---------------------------------------------------------------
# Backbone factories
# ---------------------------------------------------------------
def _load_resnet18(torch_mod):
    """Default backbone: ImageNet ResNet-18 with the FC head stripped."""
    import torchvision.models as tvm

    try:
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1
        model = tvm.resnet18(weights=weights)
    except AttributeError:  # pragma: no cover - very old torchvision
        model = tvm.resnet18(pretrained=True)

    model.fc = torch_mod.nn.Identity()
    return model, 512, (_IMAGENET_MEAN, _IMAGENET_STD)


def _load_dino(torch_mod):
    """Optional DINO ViT-S/16 backbone via torch.hub (requires network)."""
    model = torch_mod.hub.load(
        "facebookresearch/dino:main", "dino_vits16", trust_repo=True
    )
    embedding_dim = 384  # ViT-S/16 output
    return model, embedding_dim, (_IMAGENET_MEAN, _IMAGENET_STD)


def _load_clip(torch_mod):
    """Optional OpenAI CLIP image encoder; requires the ``clip`` package."""
    import clip  # type: ignore

    model, _preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    embedding_dim = int(model.visual.output_dim)

    class _CLIPWrapper(torch_mod.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, x):
            return self.clip_model.encode_image(x).float()

    return _CLIPWrapper(model), embedding_dim, (_CLIP_MEAN, _CLIP_STD)


BackboneFactory = Callable[[object], Tuple[object, int, Tuple[np.ndarray, np.ndarray]]]

# name -> (factory, human-readable description)
_BACKBONES: Dict[str, Tuple[BackboneFactory, str]] = {
    "resnet18": (_load_resnet18, "ImageNet ResNet-18 (default, 512-D)"),
    "dino": (_load_dino, "Self-supervised DINO ViT-S/16 (optional, 384-D)"),
    "clip": (_load_clip, "CLIP ViT-B/32 image encoder (optional, 512-D)"),
}


def available_backbones() -> List[str]:
    """List every backbone name known to the registry."""
    return list(_BACKBONES.keys())


class EmbeddingExtractor:
    """Produce pooled feature embeddings for 224x224 RGB frames.

    Parameters
    ----------
    device:
        Torch device spec. Defaults to CPU so the default path stays
        laptop-friendly.
    backbone:
        One of :func:`available_backbones`. Unknown names or optional
        backbones whose dependency is missing gracefully fall back to
        ``resnet18`` and the ``backbone_name`` attribute reflects the
        model actually in use.
    l2_normalize:
        If True (default), divide each output by its L2 norm. This makes
        Euclidean distance equivalent (up to a monotone transform) to
        cosine distance, which the reference-distance scorer assumes.
    """

    def __init__(
        self,
        device: str = "cpu",
        backbone: str = "resnet18",
        l2_normalize: bool = True,
    ) -> None:
        import torch  # lazy import so module import stays cheap

        self._torch = torch
        self.device = torch.device(device)
        self.l2_normalize = bool(l2_normalize)

        requested = (backbone or "resnet18").lower()
        if requested not in _BACKBONES:
            logger.warning(
                "Unknown backbone '%s'; falling back to resnet18.", backbone
            )
            requested = "resnet18"

        model, dim, stats, actual_name = self._load_with_fallback(requested)
        model.eval()
        model.to(self.device)

        self.model = model
        self.backbone_name = actual_name
        self._embedding_dim = int(dim)
        self._mean = np.asarray(stats[0], dtype=np.float32)
        self._std = np.asarray(stats[1], dtype=np.float32)

    # ---------------------------------------------------------------
    # loading
    # ---------------------------------------------------------------
    def _load_with_fallback(self, name: str):
        """Load ``name``; on any failure fall back to ResNet-18.

        Returns ``(model, embedding_dim, (mean, std), actual_name)``.
        """
        factory, _desc = _BACKBONES[name]
        try:
            model, dim, stats = factory(self._torch)
            return model, dim, stats, name
        except Exception as exc:  # pragma: no cover - depends on user env
            if name == "resnet18":
                raise
            logger.warning(
                "Optional backbone '%s' failed to load (%s). "
                "Falling back to resnet18.",
                name,
                exc,
            )
            fallback_factory, _ = _BACKBONES["resnet18"]
            model, dim, stats = fallback_factory(self._torch)
            return model, dim, stats, "resnet18"

    # ---------------------------------------------------------------
    # properties
    # ---------------------------------------------------------------
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ---------------------------------------------------------------
    # preprocessing
    # ---------------------------------------------------------------
    def _bgr_to_normalized_tensor(self, batch_bgr: np.ndarray) -> np.ndarray:
        """Convert an (N, H, W, 3) BGR uint8 array to normalized NCHW float32."""
        rgb = batch_bgr[..., ::-1]
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (0, 3, 1, 2))
        return np.ascontiguousarray(arr, dtype=np.float32)

    # ---------------------------------------------------------------
    # inference
    # ---------------------------------------------------------------
    def embed(self, images_bgr: np.ndarray) -> np.ndarray:
        """Embed a batch of BGR uint8 frames shaped ``(N, H, W, 3)``.

        Returns an ``(N, embedding_dim)`` float32 numpy array.
        """
        if images_bgr.ndim == 3:
            images_bgr = images_bgr[None, ...]
        if images_bgr.ndim != 4 or images_bgr.shape[-1] != 3:
            raise ValueError(
                f"Expected batch shape (N, H, W, 3); got {images_bgr.shape}"
            )

        torch = self._torch
        arr = self._bgr_to_normalized_tensor(images_bgr)
        tensor = torch.from_numpy(arr).to(self.device)

        with torch.inference_mode():
            feats = self.model(tensor)

        out = feats.detach().cpu().numpy().astype(np.float32, copy=False)

        if self.l2_normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            out = out / norms

        return out
