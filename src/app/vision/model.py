"""PaDiM anomaly-detection model (Defard et al., 2020).

Pretrained ResNet feature extractor -> per-patch multivariate Gaussian fit on
*defect-free* training images -> Mahalanobis distance anomaly map at inference.
No gradient training: the model is a set of per-location Gaussians, fully
deterministic and CPU-feasible.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from ..config import get_settings
from ..schemas import DefectRegion, ImageQuality, VisionResult


# ---------------- preprocessing ----------------
def _transform(size: int) -> transforms.Compose:
    # PaDiM paper: resize to 256, center-crop to 224. We normalise with ImageNet
    # stats to match the pretrained backbone.
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def validate_image(path: Path, max_mb: int) -> ImageQuality:
    """Basic upload validation: exists, decodes, not absurdly large/small."""
    if not path.exists():
        return ImageQuality(is_valid=False, reason="file not found")
    if path.stat().st_size > max_mb * 1024 * 1024:
        return ImageQuality(is_valid=False, reason=f"exceeds {max_mb} MB")
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:  # corrupt / not an image
        return ImageQuality(is_valid=False, reason=f"cannot decode image: {e}")
    if w < 32 or h < 32:
        return ImageQuality(is_valid=False, width=w, height=h,
                            reason="image too small")
    return ImageQuality(is_valid=True, width=w, height=h)


# ---------------- PaDiM core ----------------
class PaDiM:
    """PaDiM anomaly detector for a single MVTec category."""

    def __init__(self, backbone: str = "resnet18", n_features: int = 100,
                 device: str = "cpu", input_size: int = 256):
        self.settings = get_settings()
        self.backbone_name = backbone
        self.n_features = n_features
        self.device = torch.device(device)
        self.input_size = input_size
        self._tf = _transform(input_size)

        self.extractor = timm.create_model(
            backbone, pretrained=True, features_only=True
        ).to(self.device).eval()
        # resnet18 features_only -> [b,64,h/4,w/4], [b,128,h/8], [b,256,h/16]
        # PaDiM uses the first three levels upsampled to the coarsest-of-fine.

        self.means_: torch.Tensor | None = None   # (C, H*W)
        self.inv_cov_: torch.Tensor | None = None  # (C, C, H*W)
        self._inv_t: torch.Tensor | None = None   # cached (H*W, C, C) layout
        self._rand_idx: torch.Tensor | None = None
        self.normalisation_: dict = {"min": 0.0, "max": 1.0}  # score normaliser

    # -- feature extraction --
    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x)
        # take first three levels, upsample to layer1 spatial size, concat
        f1, f2, f3 = feats[0], feats[1], feats[2]
        h, w = f1.shape[-2:]
        f2 = F.interpolate(f2, size=(h, w), mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=(h, w), mode="bilinear", align_corners=False)
        emb = torch.cat([f1, f2, f3], dim=1)  # (B, C, H, W)
        if self._rand_idx is not None:
            emb = emb[:, self._rand_idx]
        return emb

    # -- fit on normal images --
    @torch.no_grad()
    def fit(self, image_paths: list[Path], batch: int = 8) -> dict:
        """Fit per-patch Gaussians on defect-free images."""
        embeddings = []
        for i in range(0, len(image_paths), batch):
            xs = []
            for p in image_paths[i:i + batch]:
                with Image.open(p) as im:
                    xs.append(self._tf(im.convert("RGB")))
            x = torch.stack(xs).to(self.device)
            embeddings.append(self._embed(x).cpu())
        emb = torch.cat(embeddings)  # (N, Ctot, H, W)
        n, ctot, h, w = emb.shape

        # random feature selection (PaDiM)
        if self._rand_idx is None:
            g = torch.Generator().manual_seed(1024)
            idx = torch.randperm(ctot, generator=g)[: self.n_features]
            self._rand_idx = idx
        emb = emb[:, self._rand_idx]  # (N, C, H, W)
        n, c, h, w = emb.shape

        emb = emb.reshape(n, c, h * w)
        mean = emb.mean(dim=0)                      # (C, HW)

        # Covariance per patch location. emb is (N, C, HW); permute to
        # (HW, N, C) so each location gives an (N, C) sample matrix, then
        # batch the covariance via einsum -> (HW, C, C). eps*I regularises.
        e = emb.permute(2, 0, 1)                    # (HW, N, C)
        e = e - e.mean(dim=1, keepdim=True)         # centre per location
        cov = torch.einsum("lnc,lnd->lcd", e, e) / (n - 1)   # (HW, C, C)
        I = torch.eye(c)
        cov = cov + 0.01 * I                        # regularise
        inv = torch.linalg.inv(cov)                 # (HW, C, C)
        # store as (C, C, HW) for the einsum in anomaly_map
        self.inv_cov_ = inv.permute(1, 2, 0).contiguous()

        self.means_ = mean
        self.spatial_ = (h, w)
        return {"n_train": n, "feature_dim": c, "map_size": (h, w)}

    # -- score one image --
    @torch.no_grad()
    def anomaly_map(self, img: Image.Image) -> np.ndarray:
        assert self.means_ is not None, "model not fitted"
        x = self._tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        emb = self._embed(x).cpu()                  # (1, C, H, W)
        c, h, w = emb.shape[1], emb.shape[2], emb.shape[3]
        emb = emb.reshape(1, c, h * w)

        delta = (emb - self.means_.unsqueeze(0))    # (1, C, HW)
        # Mahalanobis: sqrt( delta^T Sigma^-1 delta )  per location
        # einsum: for each location j, d_j @ inv[:, :, j] @ d_j
        d = delta[0]                                 # (C, HW)
        # Mahalanobis: sqrt( delta^T Sigma^-1 delta ) per location — a 2D
        # batched matmul (fast, contiguous, cached layout) instead of a naive
        # einsum, which PyTorch contracts ~1ms per location (~16s on this size).
        if self._inv_t is None:
            self._inv_t = self.inv_cov_.permute(2, 0, 1).contiguous()  # (HW,C,C)
        dt = d.permute(1, 0).contiguous()           # (HW, C)
        m2 = torch.bmm(dt.unsqueeze(1), self._inv_t).squeeze(1) * dt
        m = torch.sqrt(torch.clamp(m2.sum(dim=1), min=0))  # (HW,)
        score_map = m.reshape(h, w).numpy()

        # PaDiM paper: gaussian smoothing then upsample to input size
        score_map = gaussian_filter(score_map, sigma=4)
        score_map = np.array(
            Image.fromarray(score_map).resize(
                (self.input_size, self.input_size), Image.BILINEAR
            )
        )
        return score_map.astype(np.float32)

    def set_normalisation(self, lo: float, hi: float) -> None:
        """Score normaliser from train-good images (min-max)."""
        hi = hi if hi > lo else lo + 1e-6
        self.normalisation_ = {"min": float(lo), "max": float(hi)}

    # -- persistence --
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "backbone": self.backbone_name,
                "n_features": self.n_features,
                "input_size": self.input_size,
                "means": self.means_,
                "inv_cov": self.inv_cov_,
                "rand_idx": self._rand_idx,
                "spatial": self.spatial_,
                "normalisation": self.normalisation_,
            }, f)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "PaDiM":
        with open(path, "rb") as f:
            st = pickle.load(f)
        m = cls(backbone=st["backbone"], n_features=st["n_features"],
                device=device, input_size=st["input_size"])
        m.means_ = st["means"]
        m.inv_cov_ = st["inv_cov"]
        m._rand_idx = st["rand_idx"]
        m.spatial_ = st["spatial"]
        m.normalisation_ = st.get("normalisation", {"min": 0.0, "max": 1.0})
        return m


# ---------------- region extraction ----------------
def regions_from_map(score_map: np.ndarray, thresh: float,
                     max_regions: int) -> list[DefectRegion]:
    """Extract connected-component defect regions from a normalised map."""
    import cv2
    mask = (score_map >= thresh).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    regions: list[DefectRegion] = []
    h, w = score_map.shape
    for i in range(1, n):  # skip background
        x, y, ww, hh, area = stats[i]
        if area < 16:  # ignore specks
            continue
        region_pixels = labels == i
        mean_score = float(score_map[region_pixels].mean())
        regions.append(DefectRegion(
            region_id=i,
            bbox=(int(x), int(y), int(ww), int(hh)),
            area_px=int(area),
            area_frac=round(float(area) / (h * w), 5),
            score=round(mean_score, 4),
            centroid=(float(centroids[i][0]), float(centroids[i][1])),
        ))
    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions[:max_regions]


# ---------------- heatmap overlay ----------------
def save_overlay(image_path: Path, score_map: np.ndarray, out_dir: Path,
                 name: str) -> str:
    """Blend the anomaly score map over the original image; return filename.

    The map is min-max stretched for display only — verdicts use the raw /
    calibrated values, never this rendering.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{name}.png"
    out = out_dir / fname

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"cannot read image for overlay: {image_path}")
    img = cv2.resize(img, (score_map.shape[1], score_map.shape[0]))

    sm = score_map.astype(np.float32)
    span = float(sm.max() - sm.min())
    norm8 = ((sm - sm.min()) / (span if span > 1e-6 else 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm8, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    if not cv2.imwrite(str(out), blend):
        raise ValueError(f"failed to write overlay: {out}")
    return fname


# ---------------- calibrated threshold ----------------
def load_calibrated_threshold(model_path: Path) -> float | None:
    """Raw-score decision threshold from threshold.json next to the pickle.

    Returns None when absent/unreadable so callers fall back to the
    normalised-score threshold. Never raises.
    """
    tj = Path(model_path).parent / "threshold.json"
    if not tj.exists():
        return None
    try:
        data = json.loads(tj.read_text(encoding="utf-8"))
        v = float(data["threshold_raw"])
        return v if v > 0 else None
    except Exception:
        return None


# ---------------- high-level inspection ----------------
class AnomalyInspector:
    """Model-agnostic-ish wrapper the API/agent uses. Keeps torch out of the
    API layer and produces a structured `VisionResult`."""

    def __init__(self, model: PaDiM, category: str,
                 threshold: float | None = None,
                 calibrated_threshold: float | None = None,
                 overlay_dir: Path | None = None):
        self.model = model
        self.category = category
        s = get_settings()
        self.max_regions = s.max_defect_regions
        self.overlay_dir = overlay_dir
        if calibrated_threshold is not None:
            # Verdict on the raw score against a calibrated per-category
            # threshold; the 0-1 normalised score stays display-only.
            self.threshold = float(calibrated_threshold)
            self.verdict_mode = "calibrated_raw"
        else:
            self.threshold = threshold if threshold is not None \
                else s.anomaly_threshold
            self.verdict_mode = "normalised"

    def inspect(self, image_path: Path) -> VisionResult:
        s = get_settings()
        quality = validate_image(image_path, s.max_upload_mb)
        result = VisionResult(
            image_id=image_path.stem,
            category=self.category,
            model_name="PaDiM",
            backbone=self.model.backbone_name,
            image_anomaly_score=0.0,
            raw_anomaly_score=0.0,
            is_anomalous=False,
            threshold=self.threshold,
            verdict_mode=self.verdict_mode,
            quality=quality,
        )
        if not quality.is_valid:
            result.notes.append(f"image rejected: {quality.reason}")
            return result

        t0 = time.time()
        try:
            with Image.open(image_path) as im:
                score_map = self.model.anomaly_map(im.convert("RGB"))
        except Exception as e:
            result.notes.append(f"inference error: {e}")
            return result
        result.inference_ms = round((time.time() - t0) * 1000, 1)

        raw = float(score_map.max())
        lo, hi = self.model.normalisation_["min"], self.model.normalisation_["max"]
        norm = float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))
        result.raw_anomaly_score = round(raw, 5)
        result.image_anomaly_score = round(norm, 4)

        if self.verdict_mode == "calibrated_raw":
            # Verdict on the raw score against a calibrated per-category
            # threshold; the normalised score is display-only here.
            result.is_anomalous = bool(raw >= self.threshold)
            result.regions = regions_from_map(score_map, self.threshold,
                                              self.max_regions)
            result._score_map = score_map  # cached for overlay (not serialised)
        else:
            result.is_anomalous = bool(norm >= self.threshold)
            norm_map = np.clip((score_map - lo) / (hi - lo), 0.0, 1.0)
            result.regions = regions_from_map(norm_map, self.threshold,
                                              self.max_regions)
            result._score_map = norm_map  # cached for overlay (not serialised)

        if self.overlay_dir is not None:
            try:
                result.heatmap_path = save_overlay(
                    image_path, result._score_map, self.overlay_dir,
                    result.image_id)
            except Exception as e:
                result.notes.append(f"overlay generation failed: {e}")
        return result
