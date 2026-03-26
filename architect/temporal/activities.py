"""
temporal/activities.py -- Temporal Activities

v0.4 changes:
  execute_agent_task — now tries real Agent Connector HTTP call if endpoint_url is set.
                       Falls back to simulation if no endpoint or connector unavailable.
                       Captures structured output and passes it back through Temporal.
  notify_completion  — now also fires the callback (backup path if write_memory didn't)

Real Agent Connector call format:
  POST {endpoint_url}
  Body: StepInput (job_id, step_id, action, capability, input_data, metadata)
  Expected response: AgentOutputPacket (status, output, error, duration_ms)
"""

import asyncio
import os
from datetime import datetime, timezone

import httpx
from temporalio import activity


@activity.defn
async def execute_agent_task(agent_task: dict) -> dict:
    """
    Executes one agent task.

    If the agent has an endpoint_url (set in the registry), calls Agent Connector.
    Otherwise simulates execution (Phase 1 fallback).

    Returns a standardised result dict that matches AgentOutputPacket.
    """
    agent_id     = agent_task.get("agent_id",   "unknown")
    agent_name   = agent_task.get("agent_name", "Unknown Agent")
    step_id      = agent_task.get("step_id",    "unknown")
    action       = agent_task.get("action",     "unknown action")
    job_id       = agent_task.get("job_id",     "unknown")
    capability   = agent_task.get("capability", "general")
    sensitivity  = agent_task.get("sensitivity", "internal")
    priority     = agent_task.get("priority",   "balanced")
    input_data   = agent_task.get("input_data") or {}
    metadata     = agent_task.get("metadata",   {})
    endpoint_url = metadata.get("endpoint_url")

    activity.logger.info(
        f"Executing | job={job_id} step={step_id} agent={agent_id} action='{action}'"
    )

    started_at = datetime.now(timezone.utc).isoformat()

    # ── Real Agent Connector call ──────────────────────────────────────────
    if endpoint_url:
        result = await _call_agent_connector(
            endpoint_url = endpoint_url,
            job_id       = job_id,
            step_id      = step_id,
            agent_task_id = agent_task.get("agent_task_id", ""),
            action       = action,
            capability   = capability,
            sensitivity  = sensitivity,
            priority     = priority,
            input_data   = input_data,
            metadata     = metadata,
            started_at   = started_at,
            agent_id     = agent_id,
            agent_name   = agent_name,
        )
        activity.logger.info(
            f"Agent Connector response | step={step_id} status={result['status']} "
            f"duration={result['duration_ms']}ms"
        )
        return result

    # ── Simulation fallback (no endpoint set) ──────────────────────────────
    return await _simulate_execution(
        agent_id    = agent_id,
        agent_name  = agent_name,
        step_id     = step_id,
        action      = action,
        job_id      = job_id,
        input_data  = input_data,
        metadata    = metadata,
        started_at  = started_at,
    )


async def _call_agent_connector(
    endpoint_url:  str,
    job_id:        str,
    step_id:       str,
    agent_task_id: str,
    action:        str,
    capability:    str,
    sensitivity:   str,
    priority:      str,
    input_data:    dict,
    metadata:      dict,
    started_at:    str,
    agent_id:      str,
    agent_name:    str,
) -> dict:
    """
    Makes an HTTP POST to the Agent Connector endpoint.

    Request body (StepInput format):
      job_id, step_id, agent_task_id, action, capability,
      sensitivity, priority, input_data, metadata

    Expected response (AgentOutputPacket format):
      status, output, error, started_at, completed_at, duration_ms

    Falls back to simulation if the connector is unreachable or returns an error.
    """
    payload = {
        "job_id":        job_id,
        "step_id":       step_id,
        "agent_task_id": agent_task_id,
        "action":        action,
        "capability":    capability,
        "sensitivity":   sensitivity,
        "priority":      priority,
        "input_data":    input_data,
        "metadata":      {
            k: v for k, v in metadata.items()
            if k not in ("endpoint_url",)  # don't echo internal routing fields back
        },
    }

    timeout_secs = int(os.getenv("AGENT_CONNECTOR_TIMEOUT", "300"))

    try:
        async with httpx.AsyncClient(timeout=timeout_secs) as client:
            response = await client.post(endpoint_url, json=payload)
            response.raise_for_status()
            data = response.json()

        completed_at = datetime.now(timezone.utc).isoformat()

        # Normalise response — agent connector should return AgentOutputPacket format
        # but we handle partial responses gracefully
        output = data.get("output", data)  # if agent returns flat dict, treat it as output
        status = data.get("status", "complete")
        error  = data.get("error")

        # Compute duration if not provided
        if "duration_ms" in data:
            duration_ms = data["duration_ms"]
        else:
            try:
                from datetime import datetime as dt
                t1 = dt.fromisoformat(started_at.replace("Z", "+00:00"))
                t2 = dt.fromisoformat(completed_at.replace("Z", "+00:00"))
                duration_ms = int((t2 - t1).total_seconds() * 1000)
            except Exception:
                duration_ms = 0

        return {
            "step_id":      step_id,
            "agent_id":     agent_id,
            "agent_name":   agent_name,
            "action":       action,
            "status":       status if not error else "failed",
            "output":       output if not error else {},
            "error":        error,
            "started_at":   data.get("started_at", started_at),
            "completed_at": data.get("completed_at", completed_at),
            "duration_ms":  duration_ms,
            "source":       "agent_connector",
        }

    except httpx.HTTPStatusError as e:
        activity.logger.warning(
            f"Agent Connector HTTP error for {step_id}: {e.response.status_code} — falling back to simulation"
        )
    except httpx.RequestError as e:
        activity.logger.warning(
            f"Agent Connector unreachable for {step_id}: {e} — falling back to simulation"
        )
    except Exception as e:
        activity.logger.warning(
            f"Agent Connector unexpected error for {step_id}: {e} — falling back to simulation"
        )

    # Graceful fallback
    return await _simulate_execution(
        agent_id   = agent_id,
        agent_name = agent_name,
        step_id    = step_id,
        action     = action,
        job_id     = job_id,
        input_data = input_data,
        metadata   = metadata,
        started_at = started_at,
        source     = "simulation_fallback",
    )


async def _simulate_execution(
    agent_id:   str,
    agent_name: str,
    step_id:    str,
    action:     str,
    job_id:     str,
    input_data: dict,
    metadata:   dict,
    started_at: str,
    source:     str = "simulation",
) -> dict:
    """
    Simulates agent execution for Phase 1 / when no endpoint is available.
    Mimics realistic output structure so downstream steps can treat it as real output.
    """
    speed_score = metadata.get("agent_speed", 3)
    sim_time    = max(0.3, (6 - speed_score) * 0.3)
    await asyncio.sleep(sim_time)

    completed_at = datetime.now(timezone.utc).isoformat()
    duration_ms  = int(sim_time * 1000)

    # Build a realistic-looking output based on action type
    output = _build_simulated_output(action, agent_name, input_data)

    return {
        "step_id":      step_id,
        "agent_id":     agent_id,
        "agent_name":   agent_name,
        "action":       action,
        "status":       "complete",
        "output":       output,
        "error":        None,
        "started_at":   started_at,
        "completed_at": completed_at,
        "duration_ms":  duration_ms,
        "source":       source,
    }


def _build_simulated_output(action: str, agent_name: str, input_data: dict) -> dict:
    """
    Builds a structured simulated output that makes sense for the action type.
    Includes any input_data that was passed in, so the chain is visible.
    """
    action_lower = action.lower()

    base = {
        "summary":    f"[Simulation] {agent_name}: {action}",
        "data":       {},
        "input_received": bool(input_data),
        "input_keys":     list(input_data.keys()) if input_data else [],
    }

    if "extract" in action_lower or "parse" in action_lower:
        base["data"] = {"extracted_text": f"[Simulated extracted content for: {action}]", "pages": 1, "word_count": 250}
    elif "summar" in action_lower:
        upstream = list(input_data.values())[0] if input_data else {}
        base["data"] = {"summary": f"[Simulated summary of: {upstream.get('summary', action)}]", "bullet_points": 3}
    elif "risk" in action_lower or "analy" in action_lower:
        base["data"] = {"risk_level": "medium", "flags": [], "recommendation": f"[Simulated analysis of: {action}]"}
    elif "writ" in action_lower or "draft" in action_lower:
        base["data"] = {"content": f"[Simulated written output for: {action}]", "word_count": 150}
    elif "classif" in action_lower or "categor" in action_lower:
        base["data"] = {"category": "general", "confidence": 0.85, "tags": []}
    else:
        base["data"] = {"result": f"[Simulated output for: {action}]"}

    return base


@activity.defn
async def notify_completion(payload: dict) -> dict:
    """
    Final activity — marks job complete.
    Logs completion. Backup callback path if write_memory node didn't fire it.
    """
    job_id       = payload.get("job_id",  "unknown")
    status       = payload.get("status",  "complete")
    summary      = payload.get("summary", "")
    callback_url = payload.get("callback_url")

    activity.logger.info(f"Job complete | job={job_id} status={status} | {summary}")

    notified_at = datetime.now(timezone.utc).isoformat()

    # Fire callback if provided and not already fired by write_memory
    if callback_url:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(callback_url, json={
                    "job_id":        job_id,
                    "status":        status,
                    "summary":       summary,
                    "completed_at":  notified_at,
                })
            activity.logger.info(f"Callback fired from notify_completion -> {callback_url}")
        except Exception as e:
            activity.logger.warning(f"Callback from notify_completion failed: {e}")

    return {
        "job_id":       job_id,
        "status":       status,
        "summary":      summary,
        "notified_at":  notified_at,
    }
