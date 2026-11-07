# Architecture

## Overview

```
            ┌────────────────────────── User (web UI / API client)
            │  image + natural-language question
            ▼
┌────────────────────────────── FastAPI backend ──────────────────────────────┐
│  api/routes: /health  /inspect  /kb/ingest  /kb/query  /reports/{id}        │
│                                                                            │
│  ┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────────┐ │
│  │  validate  │──▶│ vision (PaDiM│──▶│   agent       │──▶│ report        │ │
│  │  upload    │   │ +regions)    │   │ tool-caller   │   │ generator     │ │
│  └────────────┘   └──────────────┘   └──────┬────────┘   └───────────────┘ │
│                                             │ calls                        │
│        ┌────────────────────────────────────┼─────────────────┐            │
│        ▼                ▼                   ▼                 ▼            │
│   retrieve_docs   search_history      compare_inspection   summarize/      │
│   (RAG index)     (SQLite history)    _results             generate_report │
│                                                                            │
│  ┌──────────────┐   ┌─────────────────────────────────────────────┐        │
│  │ vector index │   │ SQLite: inspections + findings (history)    │        │
│  │ (npz/json)   │   └─────────────────────────────────────────────┘        │
│  └──────────────┘                                                         │
└────────────────────────────────────────────────────────────────────────────┘
```

## Component boundaries

- **vision/** — image validation, preprocessing, PaDiM model wrapper, region
  extraction, heatmap overlay. Model-specific code is isolated behind
  `AnomalyModel` so the agent/API never touch torch directly.
- **retrieval/** — document ingestion (PDF/text → chunks with metadata),
  embedding index, hybrid search. Everything behind a `Retriever` interface.
- **database/** — SQLite persistence of inspections and findings for
  `search_inspection_history` / `compare_inspection_results`.
- **agent/** — one bounded tool-calling agent + tool registry. Tools call the
  services above and return structured `schemas` objects only.
- **reporting/** — assembles `InspectionReport` (deterministic by default).
- **api/** — thin FastAPI layer: upload validation, orchestration, serialisation.
  No business logic.

## Data flow for the MVP journey

1. `POST /inspect` with image + `question`.
2. Upload validated (type/size/decode) → `VisionResult`.
3. Agent receives question + `VisionResult`, decides tools:
   - always `inspect_image` (already run) → `retrieve_technical_documents`;
   - `search_inspection_history` if the question is comparative/historical or
     prior inspections of the category exist;
   - `compare_inspection_results` when a reference image/prior result is given;
   - `generate_inspection_report` to finish.
4. Evidence items carry kind + source; report distinguishes observations,
   retrieved facts, hypotheses and recommendations.

## Failure handling

- Bad/corrupt/oversized upload → `4xx` with a clear reason.
- Model or tool failure → caught, recorded in `trace`, reported as a
  `LIMITATION` evidence item; never silently replaced with a fake result.
- No KB indexed → retrieval returns `note="no documents indexed"` and the report
  states the knowledge base provided no supporting information.
