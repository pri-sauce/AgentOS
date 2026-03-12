"""
job_creator.py — Job Creator

Takes a TaskObject from the interpreter and creates a structured Job.
Builds steps from sub_goals and required_capabilities.
Determines formation (sequential vs parallel) based on task structure.

Phase 1: simple rule-based step building.
Later phases: LangGraph DAG with LLM-driven dependency analysis.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from models import Job, JobStep, JobStatus, StepStatus, Formation, TaskObject


def create_job(
    task_object: TaskObject,
    raw_task: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Job:
    """
    Creates a Job from a TaskObject.

    Step building logic:
    - One step per required_capability
    - If sub_goals are more granular than capabilities, use sub_goals
    - If only one step → sequential
    - If multiple steps and no natural dependency chain → parallel candidates
    - First step (extraction/parsing) is always sequential — rest can be parallel
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now    = _now()

    steps  = _build_steps(task_object)
    formation = _determine_formation(steps)

    job = Job(
        job_id      = job_id,
        session_id  = session_id,
        user_id     = user_id,
        raw_task    = raw_task,
        task_object = task_object,
        steps       = steps,
        status      = JobStatus.CREATED,
        formation   = formation,
        created_at  = now,
        updated_at  = now,
    )

    print(f"[JobCreator] Created job {job_id} with {len(steps)} step(s), formation={formation.value}")
    return job


# ── Step building ──────────────────────────────────────────────────────────

def _build_steps(task_object: TaskObject) -> List[JobStep]:
    """
    Builds job steps from the task object.

    Strategy:
    - Use required_capabilities as the primary source (one step per capability)
    - Match sub_goals to capabilities where possible for the action description
    - Mark extraction/parsing steps as must-run-first (no parallel)
    - Mark remaining steps as parallel candidates if there are 2+
    """
    capabilities = task_object.required_capabilities
    sub_goals    = task_object.sub_goals

    # Pair capabilities with sub_goals as best we can
    # If more sub_goals than capabilities — use sub_goals as steps
    # If more capabilities than sub_goals — use capabilities
    if len(sub_goals) >= len(capabilities):
        sources = sub_goals
        caps    = capabilities + ["general"] * max(0, len(sub_goals) - len(capabilities))
    else:
        sources = capabilities
        caps    = capabilities

    steps: List[JobStep] = []

    # Identify which steps are "extraction" type (must come first)
    extraction_caps = {"document_extraction", "file_parsing", "pdf_extraction", "text_extraction", "data_extraction"}
    extraction_indices = [i for i, c in enumerate(caps) if c.lower() in extraction_caps]
    has_extraction = len(extraction_indices) > 0

    for i, (source, cap) in enumerate(zip(sources, caps)):
        step_id = f"step_{i+1:02d}"
        is_extraction = cap.lower() in extraction_caps

        # Dependencies: if there's an extraction step and this is not it,
        # this step depends on the extraction step
        depends_on: List[str] = []
        if has_extraction and not is_extraction:
            extraction_step_ids = [f"step_{idx+1:02d}" for idx in extraction_indices]
            depends_on = extraction_step_ids

        # can_parallel: True if not an extraction step and there are 2+ non-extraction steps
        non_extraction_count = sum(1 for c in caps if c.lower() not in extraction_caps)
        can_parallel = (not is_extraction) and (non_extraction_count > 1)

        step = JobStep(
            step_id             = step_id,
            action              = _clean_action(source),
            required_capability = cap.lower().strip(),
            depends_on          = depends_on,
            can_parallel        = can_parallel,
            estimated_risk      = _estimate_risk(cap, task_object.sensitivity.value),
            status              = StepStatus.PENDING,
            assigned_agent_id   = None,  # filled by agent_assigner
        )
        steps.append(step)

    return steps


def _determine_formation(steps: List[JobStep]) -> Formation:
    """
    Determines whether the job runs sequentially or has parallel parts.
    If any step is marked can_parallel → formation is parallel.
    """
    if any(s.can_parallel for s in steps):
        return Formation.PARALLEL
    return Formation.SEQUENTIAL


def _estimate_risk(capability: str, sensitivity: str) -> str:
    """
    Simple risk estimate based on capability and sensitivity.
    Phase 1: rule-based. Later: LLM-reasoned.
    """
    high_risk_caps = {"risk_analysis", "compliance_check", "legal_risk", "restricted"}
    cap_lower = capability.lower()

    if sensitivity in ("restricted", "confidential") and cap_lower in high_risk_caps:
        return "high"
    if sensitivity == "confidential":
        return "medium"
    if cap_lower in high_risk_caps:
        return "medium"
    return "low"


def _clean_action(raw: str) -> str:
    """Clean up a sub_goal or capability string into a readable action description."""
    return raw.strip().replace("_", " ").lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
