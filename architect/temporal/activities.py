"""
temporal/activities.py -- Temporal Activities

Each activity is one unit of durable work.
Temporal tracks which activities completed. If the process crashes,
Temporal re-runs only the activities that didn't finish.

Activities defined here:
  execute_agent_task   -- simulates sending one agent task to the Agent Connector
                          Phase 1: simulation with status tracking
                          Later phase: real HTTP call to Agent Connector

Key Temporal guarantees per activity:
  - Retried automatically on failure (configurable retry policy)
  - State is checkpointed after each activity completes
  - Timeout enforced (activity doesn't hang forever)
"""

import asyncio
import random
from datetime import datetime, timezone

from temporalio import activity

from models import AgentTask


@activity.defn
async def execute_agent_task(agent_task: dict) -> dict:
    """
    Executes one agent task.

    Phase 1: simulation -- sleeps briefly to mimic work, returns mock result.
    Later phase: POST to Agent Connector with the instruction packet,
                 wait for result, return structured output.

    Input:  agent_task dict (serialised AgentTask)
    Output: result dict with status, output, timing
    """
    agent_id   = agent_task.get("agent_id",   "unknown")
    agent_name = agent_task.get("agent_name", "Unknown Agent")
    step_id    = agent_task.get("step_id",    "unknown")
    action     = agent_task.get("action",     "unknown action")
    job_id     = agent_task.get("job_id",     "unknown")

    activity.logger.info(
        f"Executing agent task | job={job_id} step={step_id} "
        f"agent={agent_id} action='{action}'"
    )

    started_at = datetime.now(timezone.utc).isoformat()

    # -- Phase 1 simulation --
    # Simulate variable execution time based on agent speed score
    speed_score = agent_task.get("metadata", {}).get("agent_speed", 3)
    sim_time    = max(0.5, (6 - speed_score) * 0.4)  # faster agents = shorter sim
    await asyncio.sleep(sim_time)

    # Simulate occasional failures (10% chance) to test retry/replan
    if random.random() < 0.10:
        raise Exception(
            f"Simulated agent failure | agent={agent_id} step={step_id}"
        )

    completed_at  = datetime.now(timezone.utc).isoformat()
    duration_ms   = int(sim_time * 1000)

    result = {
        "step_id":       step_id,
        "agent_id":      agent_id,
        "agent_name":    agent_name,
        "action":        action,
        "status":        "complete",
        "output":        {
            "summary": f"[Phase 1 simulation] {agent_name} completed: {action}",
            "data":    {},
        },
        "started_at":    started_at,
        "completed_at":  completed_at,
        "duration_ms":   duration_ms,
    }

    activity.logger.info(f"Activity complete | step={step_id} duration={duration_ms}ms")
    return result


@activity.defn
async def notify_completion(job_id: str, status: str, summary: str) -> dict:
    """
    Final activity -- marks the job as complete.
    Phase 1: just logs.
    Later phase: sends completion event to Event Manager.
    """
    activity.logger.info(f"Job complete | job={job_id} status={status}")
    return {
        "job_id":       job_id,
        "status":       status,
        "summary":      summary,
        "notified_at":  datetime.now(timezone.utc).isoformat(),
    }
