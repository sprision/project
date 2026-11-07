"""Report generator.

Default is a deterministic, extractive generator that assembles the structured
`InspectionReport` directly from evidence items — every claim cites its source
and uncertainty is preserved. If an LLM provider is configured in `.env`, the
same evidence is rendered by the model under an explicit grounding instruction
(never invented); the deterministic path is always available for tests.
"""
from __future__ import annotations

from ..config import get_settings
from ..schemas import (EvidenceItem, EvidenceKind, InspectionReport,
                       RetrievalResult, VisionResult)


class ReportGenerator:
    def __init__(self) -> None:
        self.s = get_settings()

    # ---------------- public ----------------
    def generate(self, image_id: str, question: str,
                 vision: VisionResult | None,
                 retrieval: RetrievalResult | dict | None,
                 evidence: list[EvidenceItem],
                 trace: list[dict]) -> InspectionReport:
        if isinstance(retrieval, dict):
            retrieval = RetrievalResult(**retrieval)
        if self.s.llm_provider in {"openai", "gemini"}:
            try:
                return self._llm_report(image_id, question, vision, retrieval,
                                        evidence, trace)
            except Exception as e:
                # Fall back to deterministic but SAY so — never silently.
                rep = self._deterministic(image_id, question, vision, retrieval,
                                          evidence, trace)
                rep.limitations.append(
                    f"LLM report generation failed ({e}); used deterministic "
                    "generator instead.")
                return rep
        return self._deterministic(image_id, question, vision, retrieval,
                                   evidence, trace)

    # ---------------- deterministic ----------------
    def _deterministic(self, image_id, question, vision, retrieval,
                       evidence, trace) -> InspectionReport:
        if isinstance(retrieval, dict):
            retrieval = RetrievalResult(**retrieval)
        findings: list[EvidenceItem] = []
        recs: list[EvidenceItem] = []
        limits: list[str] = []

        for e in evidence:
            item = EvidenceItem(**e) if isinstance(e, dict) else e
            if item.kind in (EvidenceKind.VISUAL_OBSERVATION,
                          EvidenceKind.RETRIEVED_FACT,
                          EvidenceKind.INSPECTION_HISTORY,
                          EvidenceKind.HYPOTHESIS):
                findings.append(item)
            elif item.kind == EvidenceKind.LIMITATION:
                limits.append(item.statement)

        if vision is not None:
            if vision.is_anomalous:
                if vision.verdict_mode == "calibrated_raw":
                    det = (f"raw score {vision.raw_anomaly_score:.2f} ≥ "
                           f"calibrated threshold {vision.threshold}")
                else:
                    det = (f"score {vision.image_anomaly_score:.2f} ≥ "
                           f"{vision.threshold}")
                summary = (f"Anomaly detected in {vision.image_id} "
                           f"({det}); {len(vision.regions)} "
                           "candidate defect region(s).")
                recs.append(EvidenceItem(
                    kind=EvidenceKind.RECOMMENDATION,
                    statement="Flag for human review and physical inspection of "
                              "the highlighted region(s); compare against the "
                              "acceptance criteria in the retrieved documents.",
                    source="system"))
            else:
                if vision.verdict_mode == "calibrated_raw":
                    det = (f"raw score {vision.raw_anomaly_score:.2f} < "
                           f"calibrated threshold {vision.threshold}")
                else:
                    det = (f"score {vision.image_anomaly_score:.2f} < "
                           f"{vision.threshold}")
                summary = (f"No significant anomaly detected in "
                           f"{vision.image_id} (score {det}).")
                recs.append(EvidenceItem(
                    kind=EvidenceKind.RECOMMENDATION,
                    statement="Continue routine monitoring; re-inspect on the "
                              "normal schedule or if operating conditions "
                              "change.", source="system"))
        else:
            summary = ("Image could not be analysed; see limitations. "
                       "No defect determination was made.")
            recs.append(EvidenceItem(
                kind=EvidenceKind.RECOMMENDATION,
                statement="Re-upload a valid, in-focus image of the equipment "
                          "surface and re-run the inspection.", source="system"))

        if retrieval is not None:
            chunks = getattr(retrieval, "chunks", None)
            if chunks is None and isinstance(retrieval, dict):
                chunks = retrieval.get("chunks", [])
            if not chunks:
                limits.append("No supporting passages were retrieved from the "
                              "knowledge base; recommendations are based on the "
                              "visual result only.")

        if not limits:
            limits.append("Deterministic generator; scores are model outputs, "
                          "not calibrated defect probabilities.")

        return InspectionReport(
            image_id=image_id, question=question, summary=summary,
            findings=findings, recommendations=recs, limitations=limits,
            vision=vision, retrieval=retrieval, generator="deterministic",
            trace=trace)

    # ---------------- optional LLM ----------------
    def _llm_report(self, image_id, question, vision, retrieval,
                    evidence, trace) -> InspectionReport:
        # Base report supplies structure + grounding; the LLM only rewrites the
        # summary using cited evidence. Keeps citations exact and prevents the
        # model inventing sources.
        base = self._deterministic(image_id, question, vision, retrieval,
                                   evidence, trace)
        prompt = self._build_prompt(question, vision, evidence)
        text = self._call_llm(prompt)
        if text:
            base.summary = text.strip()
            base.generator = self.s.llm_provider
        return base

    def _build_prompt(self, question, vision, evidence) -> str:
        lines = ["You are an industrial-inspection assistant. Using ONLY the "
                 "evidence below, write a 2-4 sentence summary answering the "
                 "question. Do not invent sources, values or defects. "
                 "Distinguish observation from recommendation.",
                 f"\nQuestion: {question}"]
        if vision:
            lines.append(f"Vision: score={vision.image_anomaly_score}, "
                         f"anomalous={vision.is_anomalous}, "
                         f"regions={len(vision.regions)}")
        lines.append("\nEvidence:")
        for e in evidence[:12]:
            src = f" [{e.source}]" if e.source else ""
            lines.append(f"- ({e.kind.value}) {e.statement}{src}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        if self.s.llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.s.openai_api_key)
            r = client.chat.completions.create(
                model=self.s.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=200)
            return r.choices[0].message.content or ""
        if self.s.llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.s.gemini_api_key)
            m = genai.GenerativeModel("gemini-1.5-flash")
            return (m.generate_content(prompt).text or "")
        return ""
