"""PatchCore Anomaly Detection Model (Experimental Alternative).

Implements PatchCore (Roth et al., CVPR 2022) with neighborhood-aware patch
feature extraction from intermediate ResNet-18 layers and a position-invariant
memory bank with nearest-neighbor anomaly scoring.

Designed as an experimental alternative for categories with unconstrained
spatial rotation or non-rigid deformations where PaDiM's fixed-coordinate
spatial Gaussian assumption exhibits high false positive rates.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..config import get_settings


class PatchCore:
    """PatchCore anomaly detector for industrial inspection."""

    backbone_name = "resnet18"

    def __init__(
        self,
        backbone: str = "resnet18",
        device: str = "cpu",
        coreset_size: int = 10000,
        input_size: int = 224,
    ) -> None:
        self.settings = get_settings()
        self.device = torch.device(device)
        self.coreset_size = coreset_size
        self.input_size = input_size

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.extractor = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=(2, 3)
        ).to(self.device).eval()

        self.memory_bank_: torch.Tensor | None = None  # (M, C)
        self.normalisation_: dict = {"min": 0.0, "max": 1.0}

    @torch.no_grad()
    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract neighborhood-aware patch embeddings from layer2 and layer3."""
        feats = self.extractor(x)
        f1, f2 = feats[0], feats[1]  # layer2: (B, 128, H1, W1), layer3: (B, 256, H2, W2)
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f_cat = torch.cat([f1, f2_up], dim=1)  # (B, 384, H, W)
        f_pooled = F.avg_pool2d(f_cat, kernel_size=3, stride=1, padding=1)
        return f_pooled

    @torch.no_grad()
    def fit(self, image_paths: list[Path], batch_size: int = 32, seed: int = 42) -> dict:
        """Extract patch features from defect-free training images and construct memory bank."""
        t0 = time.time()
        patch_list = []
        batch_tensors = []

        for i, p in enumerate(image_paths):
            with Image.open(p) as im:
                batch_tensors.append(self.transform(im.convert("RGB")))

            if len(batch_tensors) == batch_size or i == len(image_paths) - 1:
                batch = torch.stack(batch_tensors, dim=0).to(self.device)
                f_pooled = self._extract_patches(batch)
                B, C, H, W = f_pooled.shape
                patches = f_pooled.permute(0, 2, 3, 1).reshape(-1, C)
                patch_list.append(patches.cpu())
                batch_tensors = []

        all_patches = torch.cat(patch_list, dim=0)
        n_total_patches = all_patches.shape[0]

        # Subsample to coreset_size
        rng = np.random.default_rng(seed)
        sample_size = min(self.coreset_size, n_total_patches)
        sub_indices = rng.choice(n_total_patches, size=sample_size, replace=False)
        self.memory_bank_ = all_patches[sub_indices].to(self.device)

        elapsed = round(time.time() - t0, 2)
        return {
            "n_train_images": len(image_paths),
            "n_total_patches": n_total_patches,
            "memory_bank_size": sample_size,
            "feature_dim": int(all_patches.shape[1]),
            "elapsed_seconds": elapsed,
        }

    @torch.no_grad()
    def anomaly_map(self, img: Image.Image) -> np.ndarray:
        """Compute pixel-level anomaly score map and nearest-neighbor patch distance."""
        assert self.memory_bank_ is not None, "PatchCore model not fitted"
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        f_pooled = self._extract_patches(x)
        B, C, H, W = f_pooled.shape
        test_patches = f_pooled.permute(0, 2, 3, 1).reshape(-1, C)

        # Vectorized Euclidean nearest-neighbor distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy^T
        x_norm = (test_patches ** 2).sum(dim=1, keepdim=True)
        y_norm = (self.memory_bank_ ** 2).sum(dim=1, keepdim=True).T
        dists = torch.clamp(x_norm + y_norm - 2.0 * torch.matmul(test_patches, self.memory_bank_.T), min=0.0)
        min_dists, _ = torch.min(dists, dim=1)
        patch_scores = torch.sqrt(min_dists)

        score_map = patch_scores.reshape(1, 1, H, W)
        score_map_up = F.interpolate(
            score_map, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        )
        return score_map_up.squeeze().cpu().numpy().astype(np.float32)

    def set_normalisation(self, lo: float, hi: float) -> None:
        hi = hi if hi > lo else lo + 1e-6
        self.normalisation_ = {"min": float(lo), "max": float(hi)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "memory_bank": self.memory_bank_.cpu() if self.memory_bank_ is not None else None,
            "normalisation": self.normalisation_,
            "coreset_size": self.coreset_size,
            "input_size": self.input_size,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> PatchCore:
        state = torch.load(path, map_location=device)
        model = cls(device=device, coreset_size=state.get("coreset_size", 10000), input_size=state.get("input_size", 224))
        if state.get("memory_bank") is not None:
            model.memory_bank_ = state["memory_bank"].to(model.device)
        model.normalisation_ = state.get("normalisation", {"min": 0.0, "max": 1.0})
        return model
