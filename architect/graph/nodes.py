"""
graph/nodes.py -- LangGraph Pipeline Nodes

v0.4 changes:
  node_execute    — collects agent_results from Temporal, injects prior step
                    outputs as input_data for downstream steps
  node_aggregate  — assembles real final_result from agent_results
  node_replan     — LLM-driven replanning using Ollama with failure context
  node_write_memory — persists to PostgreSQL + pushes callback
"""

import os
import json
import uuid
from datetime import datetime, timezone

from graph.state import PipelineState
from interpreter import interpret_task
from job_creator import create_job
from agent_assigner import assign_agents
from exporter import export


# ── Node 1: Task Interpreter ───────────────────────────────────────────────

def node_interpret(state: PipelineState) -> dict:
    print(f"\n[Graph] NODE: interpret | task='{state['raw_task'][:60]}...'")
    try:
        task_object = interpret_task(
            raw_task   = state["raw_task"],
            session_id = state["session_id"],
        )
    except Exception as e:
        return {"error": f"Interpretation failed: {e}", "pipeline_done": True}

    if task_object.ambiguous:
        print(f"[Graph] Task is ambiguous -> short-circuit")
        return {
            "task_object":    task_object,
            "should_clarify": True,
            "pipeline_done":  True,
        }

    return {"task_object": task_object}


# ── Node 2: Routing Memory Check ──────────────────────────────────────────

def node_routing_check(state: PipelineState) -> dict:
    """
    Phase 1: always cache_hit=False.
    Phase 2: query pgvector here for semantic similarity hit.
    """
    print(f"[Graph] NODE: routing_check")
    return {
        "cache_hit":         False,
        "few_shot_examples": [],
    }


# ── Node 3: Plan + Job Creator ────────────────────────────────────────────

def node_plan(state: PipelineState) -> dict:
    print(f"[Graph] NODE: plan")
    try:
        job = create_job(
            task_object = state["task_object"],
            raw_task    = state["raw_task"],
            session_id  = state["session_id"],
            user_id     = state["user_id"],
        )
    except Exception as e:
        return {"error": f"Job creation failed: {e}", "pipeline_done": True}

    return {"job": job}


# ── Node 4: Agent Assigner ────────────────────────────────────────────────

def node_assign(state: PipelineState) -> dict:
    print(f"[Graph] NODE: assign | steps={len(state['job'].steps)}")
    try:
        agent_tasks = assign_agents(state["job"])
    except Exception as e:
        return {"error": f"Agent assignment failed: {e}", "pipeline_done": True}

    return {"agent_tasks": agent_tasks}


# ── Node 5: Execute (via Temporal) ────────────────────────────────────────

def node_execute(state: PipelineState) -> dict:
    """
    Submits job to Temporal. Before submitting, injects prior step outputs
    into each AgentTask.input_data so agents get context from upstream steps.

    After execution, extracts per-step results into agent_results dict.
    """
    import asyncio
    import nest_asyncio
    from temporal.workflows import run_job_workflow

    nest_asyncio.apply()

    job         = state["job"]
    agent_tasks = state["agent_tasks"]
    agent_results_so_far = state.get("agent_results", {})

    print(f"[Graph] NODE: execute | job={job.job_id} steps={len(agent_tasks)}")

    # Inject prior step outputs as input_data for each task
    _inject_prior_outputs(agent_tasks, agent_results_so_far, job)

    try:
        loop   = asyncio.get_event_loop()
        result = loop.run_until_complete(
            run_job_workflow(
                job         = job,
                agent_tasks = agent_tasks,
            )
        )

        # Extract per-step results and merge into agent_results
        new_agent_results = dict(agent_results_so_far)
        for step_result in result.get("steps_run", []):
            step_id = step_result.get("step_id")
            if step_id:
                new_agent_results[step_id] = step_result.get("output", {})
                print(f"[Graph] Step result captured: {step_id} -> {list(step_result.get('output', {}).keys())}")

        return {
            "execution_result": result,
            "agent_results":    new_agent_results,
        }

    except Exception as e:
        print(f"[Graph] Execution failed: {e}")
        return {
            "error":         f"Execution failed: {e}",
            "replan_reason": str(e),
            "should_replan": state["replan_count"] < 3,
            "replan_count":  state["replan_count"] + 1,
        }


def _inject_prior_outputs(agent_tasks: list, agent_results: dict, job) -> None:
    """
    For each agent task, if its step has input_from_steps defined,
    collects those prior step outputs and puts them in task.input_data.
    """
    step_map = {step.step_id: step for step in job.steps}

    for task in agent_tasks:
        step = step_map.get(task.step_id)
        if not step:
            continue

        input_from = getattr(step, "input_from_steps", []) or []

        if not input_from:
            # If depends_on is set but input_from_steps isn't, auto-inject all prior outputs
            input_from = step.depends_on or []

        if input_from:
            collected = {}
            for src_step_id in input_from:
                if src_step_id in agent_results:
                    collected[src_step_id] = agent_results[src_step_id]

            if collected:
                task.input_data = collected
                print(f"[Graph] Injected inputs for {task.step_id}: from steps {list(collected.keys())}")


# ── Node 6: Replan (LLM-driven) ───────────────────────────────────────────

def node_replan(state: PipelineState) -> dict:
    """
    LLM-driven replanning using Ollama.
    Sends the original plan + failure reason to the LLM and asks for a revised plan.
    Falls back to simple re-assign if LLM call fails.
    """
    print(f"[Graph] NODE: replan | attempt={state['replan_count']}")

    if state["replan_count"] >= 3:
        return {
            "error":         "Max replan attempts reached",
            "pipeline_done": True,
            "should_replan": False,
        }

    job           = state["job"]
    failure_reason = state.get("replan_reason", "Unknown failure")
    agent_tasks   = state.get("agent_tasks", [])

    # Try LLM-driven replan first
    revised_plan = _llm_replan(job, agent_tasks, failure_reason)

    if revised_plan:
        print(f"[Graph] LLM replan succeeded: {revised_plan.get('reasoning', '')[:100]}")
        # Apply revised agent assignments to existing steps
        try:
            new_agent_tasks = _apply_replan(job, revised_plan)
            return {
                "agent_tasks":   new_agent_tasks,
                "should_replan": False,
                "error":         None,
                "replan_reason": None,
            }
        except Exception as e:
            print(f"[Graph] Failed to apply LLM replan: {e}, falling back to re-assign")

    # Fallback: simple re-assign
    try:
        new_agent_tasks = assign_agents(job)
        return {
            "agent_tasks":   new_agent_tasks,
            "should_replan": False,
            "error":         None,
            "replan_reason": None,
        }
    except Exception as e:
        return {
            "error":         f"Replan failed: {e}",
            "pipeline_done": True,
        }


def _llm_replan(job, agent_tasks: list, failure_reason: str) -> dict | None:
    """
    Calls Ollama to produce a revised assignment plan based on failure context.
    Returns a dict with revised_assignments list, or None on failure.
    """
    import requests

    ollama_host  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")

    steps_summary = "\n".join(
        f"  - step_id={s.step_id}, action={s.action}, capability={s.required_capability}"
        for s in job.steps
    )

    current_assignments = "\n".join(
        f"  - step_id={t.step_id}, agent_id={t.agent_id}, agent_name={t.agent_name}"
        for t in agent_tasks
    )

    prompt = f"""You are an intelligent orchestrator replanning a failed job.

ORIGINAL TASK: {job.task_object.goal}

JOB STEPS:
{steps_summary}

CURRENT AGENT ASSIGNMENTS (that failed):
{current_assignments}

FAILURE REASON: {failure_reason}

Your job is to suggest revised agent assignments or identify which steps should be skipped/retried.

Respond ONLY with a JSON object in this exact format:
{{
  "reasoning": "brief explanation of what went wrong and what to change",
  "revised_assignments": [
    {{"step_id": "step_1", "action": "keep", "note": "this step was fine"}},
    {{"step_id": "step_2", "action": "reassign", "preferred_capability": "summarisation", "note": "original agent may be unavailable"}}
  ]
}}

Actions: "keep" | "reassign" | "skip"
Do not include any text outside the JSON object."""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)

    except Exception as e:
        print(f"[Graph] LLM replan call failed: {e}")
        return None


def _apply_replan(job, revised_plan: dict) -> list:
    """
    Applies the LLM's revised assignment plan.
    Steps with action='reassign' get a fresh agent pick.
    Steps with action='skip' are removed from the task list.
    Steps with action='keep' are re-assigned with the same capability.
    """
    from agent_registry import pick_best_agent
    from models import AgentTask, StepStatus
    from datetime import datetime, timezone

    assignments = revised_plan.get("revised_assignments", [])
    action_map  = {a["step_id"]: a for a in assignments}

    new_tasks = []
    for step in job.steps:
        directive = action_map.get(step.step_id, {})
        action    = directive.get("action", "keep")

        if action == "skip":
            print(f"[Graph] Replan: skipping step {step.step_id}")
            continue

        capability = (
            directive.get("preferred_capability")
            or step.required_capability
        )

        agent = pick_best_agent(
            capability  = capability,
            priority    = job.task_object.constraints.priority.value,
            sensitivity = job.task_object.sensitivity.value,
        )

        task = AgentTask(
            agent_task_id = f"at_{uuid.uuid4().hex[:12]}",
            job_id        = job.job_id,
            step_id       = step.step_id,
            agent_id      = agent["agent_id"],
            agent_name    = agent["name"],
            capability    = capability,
            action        = step.action,
            priority      = job.task_object.constraints.priority,
            sensitivity   = job.task_object.sensitivity,
            depends_on    = step.depends_on,
            can_parallel  = step.can_parallel,
            status        = StepStatus.PENDING,
            created_at    = datetime.now(timezone.utc).isoformat(),
            metadata      = {
                "replan_note": directive.get("note", ""),
                "replan_action": action,
                "agent_trust":   agent.get("trust_level"),
            }
        )
        new_tasks.append(task)
        print(f"[Graph] Replan: {step.step_id} [{action}] -> {agent['name']}")

    return new_tasks


# ── Node 7: Result Aggregator ─────────────────────────────────────────────

def node_aggregate(state: PipelineState) -> dict:
    """
    Assembles the final result from:
    - execution_result (Temporal workflow summary)
    - agent_results (per-step outputs collected during execute)

    Builds a structured output dict that downstream components (UI, Interaction Adapter) can use.
    """
    print(f"[Graph] NODE: aggregate")

    execution_result = state.get("execution_result", {})
    agent_results    = state.get("agent_results", {})
    job              = state.get("job")

    steps_run  = execution_result.get("steps_run", [])
    agents_used = execution_result.get("agents_used", [])

    # Build per-step output summary
    step_outputs = {}
    for step_result in steps_run:
        step_id = step_result.get("step_id")
        if step_id:
            step_outputs[step_id] = {
                "agent_id":    step_result.get("agent_id"),
                "agent_name":  step_result.get("agent_name"),
                "action":      step_result.get("action"),
                "status":      step_result.get("status", "complete"),
                "output":      step_result.get("output", {}),
                "duration_ms": step_result.get("duration_ms", 0),
            }

    # Find the last step's output as the "primary" result
    primary_output = {}
    if job and job.steps:
        last_step_id = job.steps[-1].step_id
        primary_output = agent_results.get(last_step_id, {})

    final_result = {
        "job_id":         job.job_id if job else "unknown",
        "session_id":     state.get("session_id"),
        "user_id":        state.get("user_id"),
        "raw_task":       state.get("raw_task"),
        "goal":           job.task_object.goal if job else "",
        "status":         execution_result.get("status", "complete"),
        "steps_run":      steps_run,
        "step_outputs":   step_outputs,
        "agents_used":    agents_used,
        "total_time_ms":  execution_result.get("total_time_ms", 0),
        "primary_output": primary_output,
        "all_outputs":    agent_results,
        "completed_at":   datetime.now(timezone.utc).isoformat(),
    }

    print(
        f"[Graph] Aggregated: {len(steps_run)} steps | {len(agents_used)} agents | "
        f"{execution_result.get('total_time_ms', 0)}ms"
    )

    return {"final_result": final_result, "pipeline_done": True}


# ── Node 8: Memory Writer ─────────────────────────────────────────────────

def node_write_memory(state: PipelineState) -> dict:
    """
    v0.4:
    1. Writes job + agent tasks to JSON files (existing behaviour)
    2. Persists final_result to PostgreSQL job_results table
    3. Fires callback to callback_url if provided
    """
    print(f"[Graph] NODE: write_memory | job={state['job'].job_id}")

    # 1. JSON export (existing)
    try:
        export(state["job"], state["agent_tasks"])
    except Exception as e:
        print(f"[Graph] JSON export warning: {e}")

    # 2. PostgreSQL persistence
    final_result = state.get("final_result")
    if final_result:
        try:
            from agent_registry import persist_job_result
            persist_job_result(final_result)
            print(f"[Graph] Job result persisted to PostgreSQL")
        except Exception as e:
            print(f"[Graph] PostgreSQL persist warning: {e}")

    # 3. Callback push
    callback_url = state.get("callback_url")
    if callback_url and final_result:
        _push_callback(callback_url, final_result)

    return {}


def _push_callback(callback_url: str, final_result: dict) -> None:
    """
    POSTs job result to callback_url.
    Called when Interaction Adapter or UI provided a callback_url in the TaskRequest.
    Non-blocking — failure is logged but doesn't fail the pipeline.
    """
    import requests
    try:
        payload = {
            "job_id":        final_result.get("job_id"),
            "session_id":    final_result.get("session_id"),
            "status":        final_result.get("status", "complete"),
            "summary":       f"Job completed {len(final_result.get('steps_run', []))} steps",
            "final_output":  final_result.get("primary_output", {}),
            "agents_used":   final_result.get("agents_used", []),
            "total_time_ms": final_result.get("total_time_ms", 0),
            "completed_at":  final_result.get("completed_at"),
        }
        response = requests.post(callback_url, json=payload, timeout=10)
        print(f"[Graph] Callback pushed to {callback_url} -> {response.status_code}")
    except Exception as e:
        print(f"[Graph] Callback push failed (non-fatal): {e}")


# ── Conditional edge functions ────────────────────────────────────────────

def should_clarify(state: PipelineState) -> str:
    if state.get("should_clarify") or state.get("pipeline_done"):
        return "end"
    return "routing_check"


def cache_hit_or_plan(state: PipelineState) -> str:
    if state.get("cache_hit"):
        return "execute"
    return "plan"


def after_execute(state: PipelineState) -> str:
    if state.get("should_replan"):
        return "replan"
    if state.get("pipeline_done") or state.get("error"):
        return "end"
    return "aggregate"


def after_replan(state: PipelineState) -> str:
    if state.get("pipeline_done") or state.get("error"):
        return "end"
    return "execute"
