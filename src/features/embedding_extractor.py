"""ResNet-18 feature extractor.

We load a pretrained ResNet-18, strip the classification head, and expose a
small ``embed(batch)`` helper that produces 512-D feature vectors. The model
runs on CPU in ``eval()`` mode with gradients disabled.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np


# ImageNet normalization constants (reused across torchvision models).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmbeddingExtractor:
    """Produce 512-D embeddings for 224x224 RGB frames."""

    def __init__(self, device: str = "cpu") -> None:
        # Lazy imports so ``import`` of this module is cheap.
        import torch
        import torchvision.models as tvm

        self._torch = torch
        self.device = torch.device(device)

        # torchvision >= 0.13 uses the ``weights=`` API; fall back gracefully
        # for older pins by trying both signatures.
        try:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            model = tvm.resnet18(weights=weights)
        except AttributeError:  # pragma: no cover - very old torchvision
            model = tvm.resnet18(pretrained=True)

        # Replace the classifier head with an identity so forward() returns
        # the 512-D pooled feature vector directly.
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)

        self.model = model
        self._embedding_dim = 512

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ---------------------------------------------------------------
    # preprocessing
    # ---------------------------------------------------------------
    @staticmethod
    def _bgr_to_normalized_tensor(batch_bgr: np.ndarray) -> np.ndarray:
        """Convert an (N, H, W, 3) BGR uint8 array to normalized NCHW float32."""
        # BGR -> RGB
        rgb = batch_bgr[..., ::-1]
        # uint8 -> [0, 1]
        arr = rgb.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW
        arr = np.transpose(arr, (0, 3, 1, 2))
        return np.ascontiguousarray(arr, dtype=np.float32)

    # ---------------------------------------------------------------
    # inference
    # ---------------------------------------------------------------
    def embed(self, images_bgr: np.ndarray) -> np.ndarray:
        """Embed a batch of BGR uint8 frames shaped ``(N, H, W, 3)``.

        Returns an ``(N, 512)`` float32 numpy array.
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

        return feats.detach().cpu().numpy().astype(np.float32, copy=False)

    def embed_iter(
        self,
        frames: Iterable[np.ndarray],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Embed an iterable of single BGR frames, batching internally."""
        buf: List[np.ndarray] = []
        out: List[np.ndarray] = []
        for frame in frames:
            buf.append(frame)
            if len(buf) >= batch_size:
                out.append(self.embed(np.stack(buf, axis=0)))
                buf.clear()
        if buf:
            out.append(self.embed(np.stack(buf, axis=0)))
        if not out:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        return np.concatenate(out, axis=0)
