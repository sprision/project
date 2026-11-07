"""Report generator.

Default is a deterministic, extractive generator that assembles the structured
`InspectionReport` directly from evidence items — every claim cites its source
and uncertainty is preserved. If an LLM provider is configured in `.env`, the
same evidence is rendered by the model under an explicit grounding instruction
(never invented); the deterministic path is always available for tests.
"""
from __future__ import annotations

from ..config import get_settings
from ..schemas import (CategorizedFinding, EvidenceItem, EvidenceKind,
                       FindingProvenance, InspectionReport, RetrievalResult,
                       VisionResult)


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

            # Check for borderline threshold proximity
            if vision.verdict_mode == "calibrated_raw" and vision.threshold:
                margin = abs(vision.raw_anomaly_score - vision.threshold)
                if margin < 3.0:
                    recs.append(EvidenceItem(
                        kind=EvidenceKind.RECOMMENDATION,
                        statement=f"Borderline score margin ({margin:.2f} points from threshold {vision.threshold:.2f}); "
                                  "sample lies near the decision boundary. Secondary visual verification by an inspector is advised.",
                        source="system"))
                    limits.append(f"Borderline score proximity: raw score is within {margin:.2f} points of decision threshold.")

            # Category-specific guidance & known limitations
            if vision.category == "cable":
                limits.append("Cable category exhibits documented score overlap between twisted normal samples and subtle defects; fine internal wire defects and pinprick insulation pokes require physical verification.")
                if vision.is_anomalous:
                    recs.append(EvidenceItem(
                        kind=EvidenceKind.RECOMMENDATION,
                        statement="Verify wire bundle curvature and sheath alignment manually to rule out false positive from natural cable flexure.",
                        source="system"))
            elif vision.category == "metal_nut":
                for r in vision.regions:
                    if r.bbox[0] == 0 or r.bbox[1] == 0:
                        limits.append("One or more candidate regions touch the image border; verify that elevated scores are not specular reflections or crop edge artifacts.")
                        break
            elif vision.category == "screw":
                limits.append("Screw category exhibits high pixel AUROC (0.973) for defect localization, but arbitrary screw thread rotation angles can elevate image-level anomaly scores on normal screws; verify thread defects against localized candidate regions.")
                if vision.is_anomalous:
                    recs.append(EvidenceItem(
                        kind=EvidenceKind.RECOMMENDATION,
                        statement="Examine localized bounding boxes on thread and neck regions to distinguish thread rotation from true thread stripping, bending, or head manipulation.",
                        source="system"))
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

        # Multi-detector finding categorization with strict mutual exclusivity,
        # duplicate prevention, and detector provenance tracking.
        seen_finding_ids: set[str] = set()
        structured_findings: list[CategorizedFinding] = []
        categorized_findings: dict[str, list[str]] = {
            "candidate_defects": [],
            "confirmed_visual_defects": [],  # Backwards-compatibility alias for candidate_defects
            "potential_anomalies": [],
            "uncertain_human_review": [],
        }

        def add_finding(
            finding_id: str,
            category: str,
            title: str,
            description: str,
            provenance: FindingProvenance,
            action_required: str | None = None,
        ) -> None:
            """Register a finding into EXACTLY ONE category without duplication."""
            if finding_id in seen_finding_ids:
                return
            seen_finding_ids.add(finding_id)
            cf = CategorizedFinding(
                id=finding_id,
                category=category,
                title=title,
                description=description,
                provenance=provenance,
                is_human_confirmed=False,  # Algorithmic prediction is unconfirmed until physical verification
                action_required=action_required,
            )
            structured_findings.append(cf)
            if category in categorized_findings:
                categorized_findings[category].append(description)
            if category == "candidate_defects":
                categorized_findings["confirmed_visual_defects"].append(description)

        if vision is not None:
            score = float(vision.raw_anomaly_score)
            threshold = float(vision.threshold) if vision.threshold else 0.0
            margin = score - threshold if vision.verdict_mode == "calibrated_raw" else (float(vision.image_anomaly_score) - threshold)
            is_borderline = abs(margin) < 3.0 and vision.verdict_mode == "calibrated_raw"

            has_continuity_note = any("conductor_continuity_detector" in n for n in vision.notes)
            has_puncture_note = any("surface_puncture_detector" in n for n in vision.notes)
            multi_detector_agreement = (vision.is_anomalous and (has_continuity_note or has_puncture_note)) or (has_continuity_note and has_puncture_note)

            # 1. Primary PaDiM Evaluation
            # Precedence 1: Uncertain / Human Review (|margin| < 3.0)
            if is_borderline:
                add_finding(
                    finding_id=f"padim_borderline_{vision.image_id}",
                    category="uncertain_human_review",
                    title="Borderline Score Proximity (Near Decision Boundary)",
                    description=(
                        f"PaDiM anomaly score {score:.2f} is within {abs(margin):.2f} points of decision threshold "
                        f"{threshold:.2f} ({'above' if margin >= 0 else 'below'}). Sample lies in the boundary uncertainty band."
                    ),
                    provenance=FindingProvenance(
                        detector="PaDiM", score=score, threshold=threshold, margin=round(margin, 3)
                    ),
                    action_required=(
                        "Action Required: Secondary physical verification by an inspector is advised "
                        "before approving or scrapping this component."
                    ),
                )
            elif vision.is_anomalous:
                # Precedence 2: Candidate Defect (margin >= 5.0 or multi-detector corroboration)
                if margin >= 5.0 or multi_detector_agreement:
                    add_finding(
                        finding_id=f"padim_candidate_{vision.image_id}",
                        category="candidate_defects",
                        title="Candidate Defect (Strong Visual Indication)",
                        description=(
                            f"PaDiM anomaly score {score:.2f} significantly exceeds calibrated threshold {threshold:.2f} "
                            f"(+{margin:.2f} margin) with {len(vision.regions)} candidate defect region(s)."
                        ),
                        provenance=FindingProvenance(
                            detector="PaDiM", score=score, threshold=threshold, margin=round(margin, 3)
                        ),
                        action_required="Action Required: Visually inspect candidate defect region(s) against acceptance limits.",
                    )
                # Precedence 3: Potential Anomaly (3.0 <= margin < 5.0 single detector)
                else:
                    add_finding(
                        finding_id=f"padim_potential_{vision.image_id}",
                        category="potential_anomalies",
                        title="Potential Surface Anomaly (Moderate Score Margin)",
                        description=(
                            f"PaDiM anomaly score {score:.2f} exceeds calibrated threshold {threshold:.2f} "
                            f"(+{margin:.2f} margin) with {len(vision.regions)} candidate defect region(s)."
                        ),
                        provenance=FindingProvenance(
                            detector="PaDiM", score=score, threshold=threshold, margin=round(margin, 3)
                        ),
                        action_required="Action Required: Check candidate defect region(s) on next inspection cycle.",
                    )

            # 2. Specialized Multi-Detector Evidence & Disagreement Resolution
            # Enforce strict precedence across specialized detectors:
            # Precedence 1: Uncertain / Human Review (near-boundary or detector disagreement)
            # Precedence 2: Candidate Defect (high margin or corroborated)
            # Precedence 3: Potential Anomaly (moderate single-detector indication)
            for note in vision.notes:
                if "conductor_continuity_detector" in note:
                    if is_borderline:
                        add_finding(
                            finding_id=f"cont_detector_borderline_{vision.image_id}",
                            category="uncertain_human_review",
                            title="Uncertain Finding: Conductor Discontinuity Near Decision Boundary",
                            description=f"Conductor Continuity Detector triggered near decision boundary ({score:.2f} within {abs(margin):.2f} pts of {threshold:.2f}). {note}",
                            provenance=FindingProvenance(
                                detector="ConductorContinuityDetector",
                                threshold=0.50,
                                location="wire harness divergence",
                            ),
                            action_required="Action Required: Secondary physical verification by an inspector is advised.",
                        )
                    elif not vision.is_anomalous:
                        add_finding(
                            finding_id=f"cont_detector_disagree_{vision.image_id}",
                            category="uncertain_human_review",
                            title="Detector Disagreement (Negative Anomaly Discovered): Wire Harness Conductor Discontinuity",
                            description=f"Conductor Continuity Detector discovered discontinuity while baseline PaDiM reports nominal surface ({score:.2f} < {threshold:.2f}). {note}",
                            provenance=FindingProvenance(
                                detector="ConductorContinuityDetector",
                                threshold=0.50,
                                location="wire harness divergence",
                            ),
                            action_required="Action Required: Count wire branches in divergence channels to confirm missing conductor.",
                        )
                    elif margin >= 5.0 or multi_detector_agreement:
                        add_finding(
                            finding_id=f"cont_detector_corroborated_{vision.image_id}",
                            category="candidate_defects",
                            title="Candidate Defect: Wire Harness Conductor Discontinuity",
                            description=note,
                            provenance=FindingProvenance(
                                detector="ConductorContinuityDetector",
                                threshold=0.50,
                                location="wire harness divergence",
                            ),
                            action_required="Action Required: Count wire branches in divergence channels to confirm missing conductor.",
                        )
                    else:
                        add_finding(
                            finding_id=f"cont_detector_potential_{vision.image_id}",
                            category="potential_anomalies",
                            title="Potential Anomaly: Wire Harness Conductor Discontinuity",
                            description=note,
                            provenance=FindingProvenance(
                                detector="ConductorContinuityDetector",
                                threshold=0.50,
                                location="wire harness divergence",
                            ),
                            action_required="Action Required: Re-inspect wire continuity during next cycle.",
                        )
                elif "surface_puncture_detector" in note:
                    if is_borderline:
                        add_finding(
                            finding_id=f"punc_detector_borderline_{vision.image_id}",
                            category="uncertain_human_review",
                            title="Uncertain Finding: Localized Puncture Near Decision Boundary",
                            description=f"Surface Puncture Detector flagged localized patch activation near PaDiM decision boundary ({score:.2f} within {abs(margin):.2f} pts of {threshold:.2f}). {note}",
                            provenance=FindingProvenance(
                                detector="SurfacePunctureDetector",
                                threshold=38.0,
                                location="cable jacket",
                            ),
                            action_required="Action Required: Physically verify highlighted jacket coordinates to rule out edge glints or confirm puncture.",
                        )
                    elif not vision.is_anomalous:
                        add_finding(
                            finding_id=f"punc_detector_disagree_{vision.image_id}",
                            category="uncertain_human_review",
                            title="Detector Disagreement: Localized Puncture Indication vs Nominal PaDiM",
                            description=(
                                f"Surface Puncture Detector flagged localized patch activation, while PaDiM baseline "
                                f"reports nominal overall cable geometry ({score:.2f} < {threshold:.2f}). {note}"
                            ),
                            provenance=FindingProvenance(
                                detector="SurfacePunctureDetector",
                                threshold=38.0,
                                location="cable jacket",
                            ),
                            action_required="Action Required: Examine highlighted jacket coordinates to rule out natural surface glare or flexure.",
                        )
                    elif margin >= 5.0 or multi_detector_agreement:
                        add_finding(
                            finding_id=f"punc_detector_corroborated_{vision.image_id}",
                            category="candidate_defects",
                            title="Candidate Defect: Localized Insulation Puncture",
                            description=note,
                            provenance=FindingProvenance(
                                detector="SurfacePunctureDetector",
                                threshold=38.0,
                                location="cable jacket",
                            ),
                            action_required="Action Required: Physically probe cable jacket at candidate puncture coordinates.",
                        )
                    else:
                        add_finding(
                            finding_id=f"punc_detector_potential_{vision.image_id}",
                            category="potential_anomalies",
                            title="Potential Anomaly: Localized Surface Flaw",
                            description=note,
                            provenance=FindingProvenance(
                                detector="SurfacePunctureDetector",
                                threshold=38.0,
                                location="cable jacket",
                            ),
                            action_required="Action Required: Monitor highlighted jacket coordinates on subsequent inspections.",
                        )

        # 3. Categorize remaining evidence items without duplicating
        for f in findings:
            st = f.statement
            st_low = st.lower()
            fid = f"evidence_{abs(hash(st))}"
            if "borderline" in st_low or "uncertain" in st_low:
                add_finding(
                    finding_id=fid,
                    category="uncertain_human_review",
                    title="Uncertain Finding (Requires Review)",
                    description=st,
                    provenance=FindingProvenance(detector=f.source or "InspectionAgent"),
                    action_required="Action Required: Secondary manual review recommended.",
                )
            elif "confirmed" in st_low:
                add_finding(
                    finding_id=fid,
                    category="candidate_defects",
                    title="Candidate Defect (Strong Indication)",
                    description=st,
                    provenance=FindingProvenance(detector=f.source or "InspectionAgent"),
                    action_required="Action Required: Physical inspection verification required.",
                )
            elif "anomaly" in st_low or "potential" in st_low or f.kind == EvidenceKind.HYPOTHESIS:
                add_finding(
                    finding_id=fid,
                    category="potential_anomalies",
                    title="Potential Anomaly",
                    description=st,
                    provenance=FindingProvenance(detector=f.source or "InspectionAgent"),
                    action_required="Action Required: Operational monitoring recommended.",
                )

        return InspectionReport(
            image_id=image_id, question=question, summary=summary,
            findings=findings, recommendations=recs, limitations=limits,
            vision=vision, retrieval=retrieval, generator="deterministic",
            trace=trace, structured_findings=structured_findings,
            categorized_findings=categorized_findings)

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
