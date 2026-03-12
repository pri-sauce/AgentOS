"""
temporal/workflows.py -- Temporal Workflow Definition

Temporal workflows must be deterministic -- they cannot use:
  - datetime.now()       -> use workflow.now() instead
  - random               -> not allowed in workflow code
  - asyncio.sleep        -> use workflow activities instead

All non-deterministic code lives in activities.py, not here.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.client import Client

with workflow.unsafe.imports_passed_through():
    from temporal.activities import execute_agent_task, notify_completion
    from models import Job, AgentTask


AGENT_TASK_RETRY = RetryPolicy(
    initial_interval    = timedelta(seconds=2),
    backoff_coefficient = 2.0,
    maximum_interval    = timedelta(seconds=30),
    maximum_attempts    = 3,
)


@workflow.defn
class JobWorkflow:

    @workflow.run
    async def run(self, payload: dict) -> dict:
        job_id      = payload["job"]["job_id"]
        agent_tasks = payload["agent_tasks"]
        formation   = payload["job"].get("formation", "sequential")

        workflow.logger.info(f"JobWorkflow started | job={job_id} formation={formation}")

        # Use workflow.now() -- NOT datetime.now() inside workflow code
        started_at   = workflow.now().isoformat()
        step_results = []
        agents_used  = []

        execution_groups = _group_by_dependency(agent_tasks)

        for group_index, group in enumerate(execution_groups):
            workflow.logger.info(
                f"Executing group {group_index + 1}/{len(execution_groups)} "
                f"| {len(group)} task(s) | parallel={len(group) > 1}"
            )

            if len(group) == 1:
                result = await workflow.execute_activity(
                    execute_agent_task,
                    group[0],
                    start_to_close_timeout = timedelta(minutes=5),
                    retry_policy           = AGENT_TASK_RETRY,
                )
                step_results.append(result)
                agents_used.append(group[0].get("agent_id"))

            else:
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

        total_time_ms = sum(r.get("duration_ms", 0) for r in step_results)

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
            "completed_at":  workflow.now().isoformat(),
            "output": {
                step["step_id"]: step["output"]
                for step in step_results
            },
        }


async def run_job_workflow(job: Job, agent_tasks: list) -> dict:
    """Connects to Temporal and runs the JobWorkflow. Called from LangGraph execute node."""
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
        id                = f"job-workflow-{job.job_id}",
        task_queue        = "architect-task-queue",
        execution_timeout = timedelta(minutes=30),
    )

    return result


def _group_by_dependency(agent_tasks: list[dict]) -> list[list[dict]]:
    """Groups agent tasks by execution order based on depends_on."""
    if not agent_tasks:
        return []

    completed = set()
    groups    = []
    remaining = list(agent_tasks)

    while remaining:
        ready = [
            t for t in remaining
            if all(dep in completed for dep in t.get("depends_on", []))
        ]

        if not ready:
            groups.append(remaining)
            break

        parallel_ready   = [t for t in ready if t.get("can_parallel", False)]
        sequential_ready = [t for t in ready if not t.get("can_parallel", False)]

        for t in sequential_ready:
            groups.append([t])
            completed.add(t["step_id"])

        if parallel_ready:
            groups.append(parallel_ready)
            for t in parallel_ready:
                completed.add(t["step_id"])

        remaining = [t for t in remaining if t["step_id"] not in completed]

    return groups
