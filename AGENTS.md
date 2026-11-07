# AGENTS.md

## Persistent Project Memory & Working Protocols

This repository maintains persistent project memory in the `project_context/` directory.

### Mandatory Protocols for All AI Agents

1. **Read Before Working (Orientation Protocol)**
   - Before modifying or adding any code, architecture, or documentation, every AI agent **must read all context files** in `project_context/`:
     - `project_context/MASTER_CONTEXT.md` — Project background, objectives, tech stack, and complete verified history.
     - `project_context/PROJECT_STATUS.md` — Current verified status of features, models, tests, and endpoints.
     - `project_context/ARCHITECTURE.md` — System architecture, module boundaries, schemas, and data flow.
     - `project_context/DECISIONS.md` — Key technical decisions and design trade-offs.
     - `project_context/TASK_TRACKER.md` — Completed, active, and pending tasks.
     - `project_context/BUGS_AND_ISSUES.md` — Known bugs, limitations, and open issues.
     - `project_context/SESSION_LOG.md` — Chronological log of previous session handoffs.
   - Do not make assumptions about system state; verify claims against the filesystem and execution environment.

2. **Strict Guardrails & Constraints**
   - **Do not rebuild from scratch:** This is an existing, substantially implemented project.
   - **Preserve existing models and datasets:** Do not delete or retrain models under `data/index/vision/` or overwrite `data/raw/`.
   - **Preserve evaluation runs:** Do not overwrite or fabricate results in `runs/eval_cv.json`, `runs/eval_retrieval.json`, or `runs/calibration.json`.
   - **Do not remove or alter existing functionality** without explicit user instruction.
   - **Deterministic first:** Maintain fallback deterministic operation without requiring external API keys.

3. **Update After Completing Tasks (Handoff Protocol)**
   - Upon completing any task or milestone, the agent **must update the relevant files** in `project_context/`:
     - Update `TASK_TRACKER.md` to reflect status changes (mark tasks complete, add discovered tasks).
     - Update `PROJECT_STATUS.md` with newly verified capabilities, passing tests, or components.
     - Log any newly discovered or resolved bugs in `BUGS_AND_ISSUES.md`.
     - Record a handoff entry in `SESSION_LOG.md` detailing changes made, commands run, test results, and next steps for the successor agent.
     - If architectural or design decisions were made, document them in `ARCHITECTURE.md` or `DECISIONS.md`.

