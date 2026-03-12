"""
temporal/activities.py -- Temporal Activities

Each activity is one unit of durable work.
Activities can use datetime, random, I/O -- anything non-deterministic is fine here.
Non-deterministic code must NOT go in workflows.py -- only here.

Phase 1: simulation (no real agent connector calls yet)
Later phase: real HTTP calls to Agent Connector
"""

import asyncio
from datetime import datetime, timezone

from temporalio import activity


@activity.defn
async def execute_agent_task(agent_task: dict) -> dict:
    """
    Executes one agent task.
    Phase 1: simulates work based on agent speed score.
    Later phase: POST to Agent Connector with instruction packet.
    """
    agent_id   = agent_task.get("agent_id",   "unknown")
    agent_name = agent_task.get("agent_name", "Unknown Agent")
    step_id    = agent_task.get("step_id",    "unknown")
    action     = agent_task.get("action",     "unknown action")
    job_id     = agent_task.get("job_id",     "unknown")

    activity.logger.info(
        f"Executing | job={job_id} step={step_id} agent={agent_id} action='{action}'"
    )

    started_at = datetime.now(timezone.utc).isoformat()

    # Simulate execution time based on agent speed (faster agent = less time)
    speed_score = agent_task.get("metadata", {}).get("agent_speed", 3)
    sim_time    = max(0.3, (6 - speed_score) * 0.3)
    await asyncio.sleep(sim_time)

    completed_at = datetime.now(timezone.utc).isoformat()
    duration_ms  = int(sim_time * 1000)

    result = {
        "step_id":      step_id,
        "agent_id":     agent_id,
        "agent_name":   agent_name,
        "action":       action,
        "status":       "complete",
        "output": {
            "summary": f"[Simulation] {agent_name} completed: {action}",
            "data":    {},
        },
        "started_at":   started_at,
        "completed_at": completed_at,
        "duration_ms":  duration_ms,
    }

    activity.logger.info(f"Done | step={step_id} duration={duration_ms}ms")
    return result


@activity.defn
async def notify_completion(payload: dict) -> dict:
    """
    Final activity -- marks the job as complete.
    Accepts a single dict: {job_id, status, summary}
    Phase 1: just logs. Later phase: sends event to Event Manager.
    """
    job_id  = payload.get("job_id",  "unknown")
    status  = payload.get("status",  "complete")
    summary = payload.get("summary", "")

    activity.logger.info(f"Job complete | job={job_id} status={status} | {summary}")

    return {
        "job_id":       job_id,
        "status":       status,
        "summary":      summary,
        "notified_at":  datetime.now(timezone.utc).isoformat(),
    }
