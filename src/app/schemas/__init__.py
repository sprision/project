"""Pydantic schemas exchanged between vision, retrieval, agent and API layers.

Keeping these in one place makes the contract between components explicit and
lets every layer validate the data it receives.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------- Vision ----------
class DefectRegion(BaseModel):
    region_id: int
    bbox: tuple[int, int, int, int]           # x, y, w, h in pixels
    area_px: int
    area_frac: float                          # fraction of image area
    score: float                              # mean anomaly score in region
    centroid: tuple[float, float]


class ImageQuality(BaseModel):
    is_valid: bool
    width: int | None = None
    height: int | None = None
    reason: str | None = None                 # why rejected, if is_valid=False


class VisionResult(BaseModel):
    image_id: str
    category: str | None = None               # object category model used
    model_name: str
    backbone: str
    image_anomaly_score: float                # 0..1 (normalised, for display)
    raw_anomaly_score: float                  # unnormalised model output
    is_anomalous: bool
    threshold: float
    verdict_mode: str = "normalised"          # "normalised" | "calibrated_raw"
    regions: list[DefectRegion] = Field(default_factory=list)
    quality: ImageQuality
    heatmap_path: str | None = None           # saved overlay, if produced
    inference_ms: float | None = None
    notes: list[str] = Field(default_factory=list)  # warnings / fallbacks used


# ---------- Retrieval ----------
class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    page: int | None = None
    section: str | None = None
    source: str | None = None                 # filename / URL


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    mode: str                                 # dense | bm25 | hybrid
    note: str | None = None                   # e.g. "no documents indexed"


# ---------- Agent / report ----------
class EvidenceKind(str, Enum):
    VISUAL_OBSERVATION = "visual_observation"
    RETRIEVED_FACT = "retrieved_fact"
    INSPECTION_HISTORY = "inspection_history"
    HYPOTHESIS = "hypothesis"
    RECOMMENDATION = "recommendation"
    LIMITATION = "limitation"


class EvidenceItem(BaseModel):
    kind: EvidenceKind
    statement: str
    source: str | None = None                 # doc title / tool / image region
    confidence: float | None = None           # 0..1 if applicable


class InspectionReport(BaseModel):
    image_id: str
    question: str
    summary: str
    findings: list[EvidenceItem]
    recommendations: list[EvidenceItem]
    limitations: list[str]
    vision: VisionResult | None
    retrieval: RetrievalResult | None
    generator: str                            # "deterministic" | "openai" | ...
    trace: list[dict[str, Any]] = Field(default_factory=list)  # agent tool calls
    categorized_findings: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "confirmed_visual_defects": [],
            "potential_anomalies": [],
            "uncertain_human_review": [],
        }
    )
    disclaimer: str = (
        "Decision support only. Not an authorisation to operate, repair or shut "
        "down equipment. Findings are hypotheses, not confirmed diagnoses."
    )
