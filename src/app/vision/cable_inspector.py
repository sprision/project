"""Specialized Cable Inspection Module (Multi-Detector Architecture).

Combines:
1. Production PaDiM Baseline: Detects coarse geometry & surface anomalies
   (cut outer insulation, bent wires, cable swap, combined defects).
2. Conductor Continuity Detector: Evaluates wire harness transmission channels
   to catch negative anomalies (missing wire conductors).
3. Surface Puncture Detector: Evaluates high-frequency local surface disruptions
   (multiscale patch residuals) to catch microscopic punctures (poke insulation).

Classifies findings into:
- Confirmed visual defects (high confidence / multi-detector agreement)
- Potential anomalies (single detector moderate activation)
- Uncertain findings requiring human review (borderline / negative anomaly risk)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from ..schemas import DefectRegion, VisionResult
from .model import AnomalyInspector, PaDiM, load_calibrated_threshold


@dataclass
class CableSpecializedEvidence:
    """Evidence emitted by a specialized cable inspection detector."""
    detector_name: str
    is_defect_detected: bool
    score: float
    threshold: float
    finding_type: str  # "confirmed_visual_defect", "potential_anomaly", "uncertain_human_review"
    description: str
    channel_or_location: str | None = None
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)


class ConductorContinuityDetector:
    """Specialized detector for missing wire conductors and harness discontinuities.
    
    Operates by measuring cross-sectional transmission density and edge frequency
    across the primary harness wire channels where individual conductors diverge.
    """

    CHANNEL_ROIS = {
        "top": (slice(240, 420), slice(410, 600)),
        "left": (slice(490, 690), slice(240, 450)),
        "right": (slice(490, 710), slice(580, 770)),
    }
    CHANNEL_BBOXES = {
        "top": (410, 240, 190, 180),
        "left": (240, 490, 210, 200),
        "right": (580, 490, 190, 220),
    }

    def __init__(self, threshold: float = 0.50) -> None:
        # Calibrated on seed 42 calibration subset (good max = 0.433, missing_wire min = 0.635)
        self.threshold = threshold
        self.wire_rois = dict(self.CHANNEL_ROIS)
        self.wire_bboxes = dict(self.CHANNEL_BBOXES)

    def inspect(self, img: Image.Image) -> CableSpecializedEvidence:
        arr = np.array(img.convert("RGB"))
        bg = np.array([53, 71, 87])  # MVTec AD cable background color
        diff_bg = np.linalg.norm(arr.astype(float) - bg, axis=2)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        fg_mask = (diff_bg > 35) & (gray > 30)

        # 3 primary harness wire channels: top, left, right
        channels = {
            k: fg_mask[roi[0], roi[1]] for k, roi in self.wire_rois.items()
        }
        bboxes = self.wire_bboxes

        densities = {k: float(np.mean(m)) for k, m in channels.items()}
        min_ch = min(densities, key=densities.get)
        min_density = densities[min_ch]
        defect_score = float(1.0 - min_density)

        is_detected = defect_score >= self.threshold
        if is_detected:
            margin = defect_score - self.threshold
            finding_type = "confirmed_visual_defect" if margin >= 0.10 else "potential_anomaly"
            desc = (
                f"Conductor discontinuity detected in {min_ch} wire channel "
                f"(channel transmission density {min_density:.2f} vs expected >= 0.50). "
                f"Indicates missing or broken conductor branch."
            )
        elif defect_score >= (self.threshold - 0.08):
            finding_type = "uncertain_human_review"
            desc = (
                f"Borderline conductor density in {min_ch} channel ({min_density:.2f}). "
                f"Potential partial wire dropout; secondary visual verification recommended."
            )
        else:
            finding_type = "potential_anomaly" if is_detected else "normal"
            desc = f"All conductor harness channels exhibit continuous transmission (min density {min_density:.2f})."

        return CableSpecializedEvidence(
            detector_name="conductor_continuity_detector",
            is_defect_detected=is_detected,
            score=defect_score,
            threshold=self.threshold,
            finding_type=finding_type,
            description=desc,
            channel_or_location=min_ch,
            bbox=bboxes[min_ch] if is_detected else None,
        )


class SurfacePunctureDetector:
    """Specialized detector for microscopic insulation jacket punctures (poke_insulation).
    
    Evaluates multi-scale anomaly map residuals to prevent PaDiM's default sigma=4
    Gaussian smoothing from averaging out high-frequency sub-millimeter punctures.
    """

    def __init__(self, padim_model: PaDiM | None = None, threshold: float = 38.0) -> None:
        self.model = padim_model
        # Calibrated on seed 42 calibration subset (good sigma1 avg = 28.5, poke sigma1 avg = 45.2)
        self.threshold = threshold

    @torch.no_grad()
    def inspect(self, img: Image.Image) -> CableSpecializedEvidence:
        if self.model is None or not hasattr(self.model, "means_") or self.model.means_ is None:
            return CableSpecializedEvidence(
                detector_name="surface_puncture_detector",
                is_defect_detected=False,
                score=0.0,
                threshold=self.threshold,
                finding_type="normal",
                description="Surface puncture detector inactive (no PaDiM model attached).",
                channel_or_location="n/a",
                bbox=None,
            )

        x = self.model._tf(img.convert("RGB")).unsqueeze(0).to(self.model.device)
        emb = self.model._embed(x).cpu()
        c, h, w = emb.shape[1], emb.shape[2], emb.shape[3]
        emb = emb.reshape(1, c, h * w)

        delta = emb - self.model.means_.unsqueeze(0)
        d = delta[0]
        if self.model._inv_t is None:
            self.model._inv_t = self.model.inv_cov_.permute(2, 0, 1).contiguous()
        dt = d.permute(1, 0).contiguous()
        m2 = torch.bmm(dt.unsqueeze(1), self.model._inv_t).squeeze(1) * dt
        m = torch.sqrt(torch.clamp(m2.sum(dim=1), min=0))
        raw_map = m.reshape(h, w).numpy()

        # Local high-frequency map with light smoothing (sigma=1.0)
        s1_map = gaussian_filter(raw_map, sigma=1.0)
        # Suppress outer boundary patches (2 patches ~ 16px) to eliminate framing edge noise
        s1_interior = s1_map.copy()
        s1_interior[:2, :] = 0.0
        s1_interior[-2:, :] = 0.0
        s1_interior[:, :2] = 0.0
        s1_interior[:, -2:] = 0.0

        max_s1 = float(s1_interior.max())

        # Find location of peak high-frequency score
        py, px = np.unravel_index(np.argmax(s1_interior), s1_interior.shape)
        # Convert 64x64 patch coordinates to 1024x1024 image coordinates
        scale = 1024.0 / h
        cx, cy = int(px * scale), int(py * scale)
        bbox = (max(0, cx - 40), max(0, cy - 40), 80, 80)

        is_detected = max_s1 >= self.threshold
        if is_detected:
            margin = max_s1 - self.threshold
            finding_type = "confirmed_visual_defect" if margin >= 5.0 else "potential_anomaly"
            desc = (
                f"Localized insulation puncture detected at (x={cx}, y={cy}) with high-frequency "
                f"patch score {max_s1:.2f} (exceeds jacket threshold {self.threshold:.2f})."
            )
        elif max_s1 >= (self.threshold - 4.0):
            finding_type = "uncertain_human_review"
            desc = (
                f"Borderline local patch activation at (x={cx}, y={cy}) (score {max_s1:.2f} vs {self.threshold:.2f}). "
                f"Possible microscopic jacket disruption; operator review advised."
            )
        else:
            finding_type = "normal"
            desc = f"Insulation jacket surface is smooth and intact (max local patch score {max_s1:.2f})."

        return CableSpecializedEvidence(
            detector_name="surface_puncture_detector",
            is_defect_detected=is_detected,
            score=max_s1,
            threshold=self.threshold,
            finding_type=finding_type,
            description=desc,
            channel_or_location=f"x={cx},y={cy}",
            bbox=bbox if is_detected else None,
        )


class SpecializedCableInspector:
    """Composite inspection engine integrating production PaDiM and specialized detectors."""

    def __init__(
        self,
        padim_inspector: AnomalyInspector,
        continuity_threshold: float = 0.50,
        puncture_threshold: float = 38.0,
        continuity_detector: ConductorContinuityDetector | None = None,
        puncture_detector: SurfacePunctureDetector | None = None,
    ) -> None:
        self.padim_inspector = padim_inspector
        self.continuity_detector = (
            continuity_detector
            if continuity_detector is not None
            else ConductorContinuityDetector(threshold=continuity_threshold)
        )
        if puncture_detector is not None:
            self.puncture_detector = puncture_detector
        elif hasattr(padim_inspector, "model") and padim_inspector.model is not None:
            self.puncture_detector = SurfacePunctureDetector(padim_inspector.model, threshold=puncture_threshold)
        else:
            self.puncture_detector = SurfacePunctureDetector(None, threshold=puncture_threshold)

    def inspect(self, image_input: Path | str | Image.Image) -> tuple[VisionResult, list[CableSpecializedEvidence]]:
        """Execute composite multi-detector inspection and categorize findings."""
        t0 = time.time()
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            base_result = self.padim_inspector.inspect(image_path)
        else:
            img = image_input.convert("RGB")
            base_result = self.padim_inspector.inspect(img)

        # 2. Specialized detectors
        continuity_ev = self.continuity_detector.inspect(img)
        puncture_ev = self.puncture_detector.inspect(img)

        specialized_evidence = [continuity_ev, puncture_ev]

        # 3. Evidence fusion and defect augmentation
        augmented_regions = list(base_result.regions)
        is_composite_anomalous = base_result.is_anomalous
        notes = list(base_result.notes)

        next_region_id = len(augmented_regions) + 1
        img_w, img_h = img.size
        total_px = max(img_w * img_h, 1)

        if continuity_ev.is_defect_detected and continuity_ev.bbox:
            x, y, w, h = continuity_ev.bbox
            area = w * h
            augmented_regions.append(
                DefectRegion(
                    region_id=next_region_id,
                    bbox=(x, y, w, h),
                    area_px=area,
                    area_frac=round(area / total_px, 6),
                    score=float(continuity_ev.score),
                    centroid=(float(x + w / 2), float(y + h / 2)),
                )
            )
            next_region_id += 1
            is_composite_anomalous = True
            notes.append(f"Specialized detector '{continuity_ev.detector_name}' detected conductor continuity failure in {continuity_ev.channel_or_location}.")

        if puncture_ev.is_defect_detected and puncture_ev.bbox:
            x, y, w, h = puncture_ev.bbox
            area = w * h
            augmented_regions.append(
                DefectRegion(
                    region_id=next_region_id,
                    bbox=(x, y, w, h),
                    area_px=area,
                    area_frac=round(area / total_px, 6),
                    score=float(puncture_ev.score),
                    centroid=(float(x + w / 2), float(y + h / 2)),
                )
            )
            next_region_id += 1
            is_composite_anomalous = True
            notes.append(f"Specialized detector '{puncture_ev.detector_name}' detected surface puncture (score {puncture_ev.score:.2f} >= {puncture_ev.threshold:.2f}).")

        # Sort regions by size descending
        augmented_regions.sort(key=lambda r: r.area_px, reverse=True)

        composite_result = VisionResult(
            image_id=base_result.image_id,
            category=base_result.category or "cable",
            model_name=f"{base_result.model_name}+SpecializedCableInspector",
            backbone=base_result.backbone,
            image_anomaly_score=max(base_result.image_anomaly_score, 1.0 if (continuity_ev.is_defect_detected or puncture_ev.is_defect_detected) else base_result.image_anomaly_score),
            raw_anomaly_score=max(base_result.raw_anomaly_score, puncture_ev.score if puncture_ev.is_defect_detected else base_result.raw_anomaly_score),
            is_anomalous=is_composite_anomalous,
            threshold=base_result.threshold,
            verdict_mode=base_result.verdict_mode,
            regions=augmented_regions,
            quality=base_result.quality,
            heatmap_path=base_result.heatmap_path,
            inference_ms=round((time.time() - t0) * 1000.0, 2),
            notes=notes,
        )

        return composite_result, specialized_evidence
