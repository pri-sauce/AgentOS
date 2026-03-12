"""
temporal/workflows.py -- Temporal Workflow Definition

The JobWorkflow is the durable execution unit for one job.

What Temporal gives us here:
  - Each activity (agent task) is checkpointed when complete
  - If the Python process crashes, Temporal re-runs from last checkpoint
  - Parallel agent tasks run as concurrent activities natively
  - Timeouts and retries are defined per activity
  - The workflow can be paused (e.g. for human approval) and resumed

Workflow structure:
  1. For sequential steps: run activities one by one in order
  2. For parallel steps (same depends_on, can_parallel=True): run concurrently
  3. After all activities complete: notify_completion
"""

import asyncio
from datetime import timedelta, datetime, timezone

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.client import Client

# Import activities via workflow.execute_activity (not direct import)
# This is the Temporal pattern -- activities are referenced by name
with workflow.unsafe.imports_passed_through():
    from temporal.activities import execute_agent_task, notify_completion
    from models import Job, AgentTask


# ── Retry policy for agent task activities ────────────────────────────────

AGENT_TASK_RETRY = RetryPolicy(
    initial_interval    = timedelta(seconds=2),
    backoff_coefficient = 2.0,
    maximum_interval    = timedelta(seconds=30),
    maximum_attempts    = 3,
)


# ── Workflow ──────────────────────────────────────────────────────────────

@workflow.defn
class JobWorkflow:
    """
    Durable workflow for executing one Architect job.
    Runs all agent tasks with proper sequencing and parallelism.
    """

    @workflow.run
    async def run(self, payload: dict) -> dict:
        """
        payload = {
          "job":         serialised Job dict,
          "agent_tasks": list of serialised AgentTask dicts
        }
        """
        job_id      = payload["job"]["job_id"]
        agent_tasks = payload["agent_tasks"]
        formation   = payload["job"].get("formation", "sequential")

        workflow.logger.info(f"JobWorkflow started | job={job_id} formation={formation}")

        started_at   = datetime.now(timezone.utc).isoformat()
        step_results = []
        agents_used  = []

        # -- Group tasks by their depends_on to determine execution order --
        # Steps with no dependencies run first
        # Steps that depend on prior steps run after those complete
        execution_groups = _group_by_dependency(agent_tasks)

        for group_index, group in enumerate(execution_groups):
            workflow.logger.info(
                f"Executing group {group_index + 1}/{len(execution_groups)} "
                f"| {len(group)} task(s) | parallel={len(group) > 1}"
            )

            if len(group) == 1:
                # Sequential -- run single activity
                result = await workflow.execute_activity(
                    execute_agent_task,
                    group[0],
                    start_to_close_timeout = timedelta(minutes=5),
                    retry_policy           = AGENT_TASK_RETRY,
                )
                step_results.append(result)
                agents_used.append(group[0].get("agent_id"))

            else:
                # Parallel -- run all activities in this group concurrently
                tasks = [
                    workflow.execute_activity(
                        execute_agent_task,
                        task,
                        start_to_close_timeout = timedelta(minutes=5),
                        retry_policy           = AGENT_TASK_RETRY,
                    )
                    for task in group
                ]
                results = await asyncio.gather(*tasks)
                step_results.extend(results)
                agents_used.extend([t.get("agent_id") for t in group])

        # -- Calculate total time --
        total_time_ms = sum(r.get("duration_ms", 0) for r in step_results)

        # -- Notify completion --
        await workflow.execute_activity(
            notify_completion,
            job_id,
            "complete",
            f"Job {job_id} completed {len(step_results)} steps",
            start_to_close_timeout = timedelta(seconds=10),
        )

        workflow.logger.info(f"JobWorkflow complete | job={job_id} steps={len(step_results)}")

        return {
            "job_id":        job_id,
            "status":        "complete",
            "steps_run":     step_results,
            "agents_used":   list(set(agents_used)),
            "total_time_ms": total_time_ms,
            "started_at":    started_at,
            "completed_at":  datetime.now(timezone.utc).isoformat(),
            "output":        {
                step["step_id"]: step["output"]
                for step in step_results
            },
        }


# ── Client helper ─────────────────────────────────────────────────────────

async def run_job_workflow(job: Job, agent_tasks: list[AgentTask]) -> dict:
    """
    Connects to Temporal server and starts the JobWorkflow.
    Called from the LangGraph execute node.

    Returns the workflow result dict.
    """
    import os
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")

    client = await Client.connect(temporal_host)

    payload = {
        "job":         job.model_dump(),
        "agent_tasks": [at.model_dump() for at in agent_tasks],
    }

    result = await client.execute_workflow(
        JobWorkflow.run,
        payload,
        id              = f"job-workflow-{job.job_id}",
        task_queue      = "architect-task-queue",
        execution_timeout = timedelta(minutes=30),
    )

    return result


# ── Dependency grouping helper ────────────────────────────────────────────

def _group_by_dependency(agent_tasks: list[dict]) -> list[list[dict]]:
    """
    Groups agent tasks by execution order based on depends_on.

    Returns a list of groups where:
    - Group 0: tasks with no dependencies (run first)
    - Group 1: tasks that depend on group 0 (run after group 0)
    - etc.

    Tasks within the same group run in parallel if can_parallel=True.
    """
    if not agent_tasks:
        return []

    # Build a map of step_id -> task
    task_map = {t["step_id"]: t for t in agent_tasks}

    # Topological sort into groups
    completed = set()
    groups    = []

    remaining = list(agent_tasks)

    while remaining:
        # Find tasks whose dependencies are all completed
        ready = [
            t for t in remaining
            if all(dep in completed for dep in t.get("depends_on", []))
        ]

        if not ready:
            # Circular dependency or bad data -- just run everything remaining
            groups.append(remaining)
            break

        # Split ready tasks into parallel vs sequential
        parallel_ready = [t for t in ready if t.get("can_parallel", False)]
        sequential_ready = [t for t in ready if not t.get("can_parallel", False)]

        # Sequential tasks each get their own group
        for t in sequential_ready:
            groups.append([t])
            completed.add(t["step_id"])

        # Parallel tasks all go in one group
        if parallel_ready:
            groups.append(parallel_ready)
            for t in parallel_ready:
                completed.add(t["step_id"])

        remaining = [t for t in remaining if t["step_id"] not in completed]

    return groups
