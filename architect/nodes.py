"""
graph/nodes.py -- LangGraph Pipeline Nodes

Each function here is one node in the pipeline graph.
Every node receives the full PipelineState, does its work,
and returns a dict of fields to update in the state.

LangGraph merges the returned dict back into the state automatically.

Node order:
  interpret -> routing_check -> plan -> assign -> execute -> aggregate -> write_memory

Conditional edges:
  interpret   -> if ambiguous        -> END (return clarification)
  routing_check -> if cache hit      -> skip to execute
  execute     -> if failed           -> replan (max 3x) or END with error
"""

import uuid
from datetime import datetime, timezone

from graph.state import PipelineState
from interpreter import interpret_task
from job_creator import create_job
from agent_assigner import assign_agents
from exporter import export


# ── Node 1: Task Interpreter ───────────────────────────────────────────────

def node_interpret(state: PipelineState) -> dict:
    """
    Runs the CoT + Ollama task interpreter.
    Produces a structured TaskObject.
    If ambiguous, sets should_clarify=True to short-circuit the graph.
    """
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
            "task_object":     task_object,
            "should_clarify":  True,
            "pipeline_done":   True,
        }

    return {"task_object": task_object}


# ── Node 2: Routing Memory Check ──────────────────────────────────────────

def node_routing_check(state: PipelineState) -> dict:
    """
    Phase 1: basic routing check placeholder.
    In a later phase this queries pgvector for similar past tasks,
    checks confidence, pulls few-shot examples.

    For now: always returns cache_hit=False so full pipeline runs.
    Few-shot examples are empty list (baked into CoT prompt already).
    """
    print(f"[Graph] NODE: routing_check")

    # Phase 1 -- no routing memory yet, always miss
    # Phase 2 -- query pgvector here, check confidence threshold
    return {
        "cache_hit":         False,
        "few_shot_examples": [],
    }


# ── Node 3: Plan + Job Creator ────────────────────────────────────────────

def node_plan(state: PipelineState) -> dict:
    """
    Turns the TaskObject into a Job with steps.
    Skipped if cache_hit=True (routing memory already knows the plan).
    """
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
    """
    Assigns the best agent to each step in the job.
    Queries PostgreSQL registry.
    Skipped if cache_hit=True.
    """
    print(f"[Graph] NODE: assign | steps={len(state['job'].steps)}")

    try:
        agent_tasks = assign_agents(state["job"])
    except Exception as e:
        return {"error": f"Agent assignment failed: {e}", "pipeline_done": True}

    return {"agent_tasks": agent_tasks}


# ── Node 5: Execute (via Temporal) ────────────────────────────────────────

def node_execute(state: PipelineState) -> dict:
    """
    Submits the job to Temporal for durable execution.
    Uses nest_asyncio to handle FastAPI already running an event loop.
    """
    import asyncio
    import nest_asyncio
    from temporal.workflows import run_job_workflow

    nest_asyncio.apply()

    print(f"[Graph] NODE: execute | job={state['job'].job_id}")

    try:
        loop   = asyncio.get_event_loop()
        result = loop.run_until_complete(
            run_job_workflow(
                job         = state["job"],
                agent_tasks = state["agent_tasks"],
            )
        )
        return {"execution_result": result}

    except Exception as e:
        print(f"[Graph] Execution failed: {e}")
        return {
            "error":         f"Execution failed: {e}",
            "should_replan": state["replan_count"] < 3,
            "replan_count":  state["replan_count"] + 1,
        }


# ── Node 6: Replan ────────────────────────────────────────────────────────

def node_replan(state: PipelineState) -> dict:
    """
    Called when execution fails and replan_count < 3.
    Phase 1: simple retry -- resets agent tasks and tries again.
    Later phase: LangGraph node calls LLM with Zero-Shot CoT to revise plan.
    """
    print(f"[Graph] NODE: replan | attempt={state['replan_count']}")

    if state["replan_count"] >= 3:
        return {
            "error":         "Max replan attempts reached",
            "pipeline_done": True,
            "should_replan": False,
        }

    # Phase 1: just re-assign agents (simplest replan)
    try:
        agent_tasks = assign_agents(state["job"])
        return {
            "agent_tasks":   agent_tasks,
            "should_replan": False,
            "error":         None,
        }
    except Exception as e:
        return {
            "error":         f"Replan failed: {e}",
            "pipeline_done": True,
        }


# ── Node 7: Result Aggregator ─────────────────────────────────────────────

def node_aggregate(state: PipelineState) -> dict:
    """
    Collects execution results and builds the final result object.
    Phase 1: assembles result from Temporal workflow output.
    Later phase: LLM merge for parallel step results.
    """
    print(f"[Graph] NODE: aggregate")

    execution_result = state.get("execution_result", {})

    final_result = {
        "job_id":        state["job"].job_id,
        "status":        execution_result.get("status", "complete"),
        "steps_run":     execution_result.get("steps_run", []),
        "agents_used":   execution_result.get("agents_used", []),
        "total_time_ms": execution_result.get("total_time_ms", 0),
        "output":        execution_result.get("output", {}),
        "completed_at":  datetime.now(timezone.utc).isoformat(),
    }

    return {"final_result": final_result, "pipeline_done": True}


# ── Node 8: Memory Writer ─────────────────────────────────────────────────

def node_write_memory(state: PipelineState) -> dict:
    """
    Writes job + agent tasks to JSON files (existing exporter).
    Phase 2: also writes to routing memory in PostgreSQL + pgvector.
    """
    print(f"[Graph] NODE: write_memory | job={state['job'].job_id}")

    try:
        export(state["job"], state["agent_tasks"])
    except Exception as e:
        print(f"[Graph] Memory write warning (non-fatal): {e}")

    return {}


# ── Conditional edge functions ────────────────────────────────────────────

def should_clarify(state: PipelineState) -> str:
    """After interpret: if ambiguous -> END, else continue."""
    if state.get("should_clarify") or state.get("pipeline_done"):
        return "end"
    return "routing_check"


def cache_hit_or_plan(state: PipelineState) -> str:
    """After routing check: if cache hit -> execute directly, else plan."""
    if state.get("cache_hit"):
        return "execute"
    return "plan"


def after_execute(state: PipelineState) -> str:
    """After execute: if failed and can replan -> replan, else aggregate."""
    if state.get("should_replan"):
        return "replan"
    if state.get("pipeline_done") or state.get("error"):
        return "end"
    return "aggregate"


def after_replan(state: PipelineState) -> str:
    """After replan: if still failed -> end, else retry execute."""
    if state.get("pipeline_done") or state.get("error"):
        return "end"
    return "execute"
