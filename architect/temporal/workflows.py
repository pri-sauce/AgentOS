"""
temporal/workflows.py -- Temporal Workflow Definition

v0.4 changes:
  - Passes prior step outputs as input_data to downstream tasks
    (agent_results dict built up as steps complete, injected before each group runs)
  - Passes callback_url through to notify_completion
  - Groups now execute in proper dependency order with real output chaining
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
        job_dict    = payload["job"]
        agent_tasks = payload["agent_tasks"]
        callback_url = payload.get("callback_url")

        job_id    = job_dict["job_id"]
        formation = job_dict.get("formation", "sequential")

        workflow.logger.info(f"JobWorkflow started | job={job_id} formation={formation} tasks={len(agent_tasks)}")

        started_at    = workflow.now().isoformat()
        step_results  = []
        agents_used   = []
        agent_results = {}  # step_id -> output dict, built as we go

        execution_groups = _group_by_dependency(agent_tasks)

        for group_index, group in enumerate(execution_groups):
            workflow.logger.info(
                f"Group {group_index + 1}/{len(execution_groups)} | "
                f"{len(group)} task(s) | parallel={len(group) > 1}"
            )

            # Inject accumulated prior outputs into each task in this group
            group_with_inputs = _inject_prior_outputs(group, agent_results, job_dict)

            if len(group_with_inputs) == 1:
                result = await workflow.execute_activity(
                    execute_agent_task,
                    group_with_inputs[0],
                    start_to_close_timeout = timedelta(minutes=5),
                    retry_policy           = AGENT_TASK_RETRY,
                )
                step_results.append(result)
                agents_used.append(group_with_inputs[0].get("agent_id"))
                # Store this step's output for downstream steps
                agent_results[result["step_id"]] = result.get("output", {})

            else:
                tasks = [
                    workflow.execute_activity(
                        execute_agent_task,
                        task,
                        start_to_close_timeout = timedelta(minutes=5),
                        retry_policy           = AGENT_TASK_RETRY,
                    )
                    for task in group_with_inputs
                ]
                results = await asyncio.gather(*tasks)
                step_results.extend(results)
                for r in results:
                    agents_used.append(r.get("agent_id"))
                    agent_results[r["step_id"]] = r.get("output", {})

        total_time_ms = sum(r.get("duration_ms", 0) for r in step_results)

        await workflow.execute_activity(
            notify_completion,
            {
                "job_id":        job_id,
                "status":        "complete",
                "summary":       f"Job {job_id} completed {len(step_results)} steps",
                "callback_url":  callback_url,
            },
            start_to_close_timeout = timedelta(seconds=10),
        )

        workflow.logger.info(f"JobWorkflow complete | job={job_id} steps={len(step_results)}")

        return {
            "job_id":        job_id,
            "status":        "complete",
            "steps_run":     step_results,
            "agents_used":   list(set(filter(None, agents_used))),
            "total_time_ms": total_time_ms,
            "started_at":    started_at,
            "completed_at":  workflow.now().isoformat(),
            "output": {
                step["step_id"]: step["output"]
                for step in step_results
            },
            "agent_results": agent_results,
        }


def _inject_prior_outputs(group: list[dict], agent_results: dict, job_dict: dict) -> list[dict]:
    """
    For each task in the group, build input_data from prior step results.
    Uses depends_on to know which prior steps to pull outputs from.
    Returns new list of task dicts with input_data populated.
    """
    # Build step-level input_from_steps map from job steps
    input_from_map = {}
    for step in job_dict.get("steps", []):
        step_id = step.get("step_id")
        input_from = step.get("input_from_steps") or step.get("depends_on") or []
        if step_id and input_from:
            input_from_map[step_id] = input_from

    updated = []
    for task in group:
        task = dict(task)  # don't mutate original
        step_id    = task.get("step_id", "")
        depends_on = task.get("depends_on") or input_from_map.get(step_id) or []

        if depends_on:
            prior_outputs = {}
            for src_step_id in depends_on:
                if src_step_id in agent_results:
                    prior_outputs[src_step_id] = agent_results[src_step_id]

            if prior_outputs:
                task["input_data"] = prior_outputs

        updated.append(task)
    return updated


async def run_job_workflow(job: Job, agent_tasks: list, callback_url: str = None) -> dict:
    """Connects to Temporal and runs the JobWorkflow."""
    import os
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    payload = {
        "job":          job.model_dump(),
        "agent_tasks":  [at.model_dump() for at in agent_tasks],
        "callback_url": callback_url,
    }

    return await client.execute_workflow(
        JobWorkflow.run,
        payload,
        id                = f"job-workflow-{job.job_id}",
        task_queue        = "architect-task-queue",
        execution_timeout = timedelta(minutes=30),
    )


def _group_by_dependency(agent_tasks: list[dict]) -> list[list[dict]]:
    """Groups agent tasks into execution groups based on depends_on."""
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
