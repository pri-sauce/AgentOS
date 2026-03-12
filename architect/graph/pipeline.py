"""
graph/pipeline.py -- LangGraph Pipeline Assembly

Builds and compiles the full Architect pipeline as a LangGraph StateGraph.

Graph structure:

  START
    |
  [interpret]
    |
  (conditional) -- ambiguous? --> END
    |
  [routing_check]
    |
  (conditional) -- cache hit? --> [execute]
    |                                 |
  [plan]                              |
    |                                 |
  [assign] ----------------------------
    |
  [execute]
    |
  (conditional) -- failed? replan_count < 3? --> [replan] --> [execute]
    |                                                |
    |                                               END (max replans)
    |
  [aggregate]
    |
  [write_memory]
    |
   END
"""

from langgraph.graph import StateGraph, END

from graph.state import PipelineState
from graph.nodes import (
    node_interpret,
    node_routing_check,
    node_plan,
    node_assign,
    node_execute,
    node_replan,
    node_aggregate,
    node_write_memory,
    should_clarify,
    cache_hit_or_plan,
    after_execute,
    after_replan,
)

# ── Build the graph ────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Assembles and compiles the full Architect pipeline graph.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(PipelineState)

    # -- Add nodes --
    graph.add_node("interpret",      node_interpret)
    graph.add_node("routing_check",  node_routing_check)
    graph.add_node("plan",           node_plan)
    graph.add_node("assign",         node_assign)
    graph.add_node("execute",        node_execute)
    graph.add_node("replan",         node_replan)
    graph.add_node("aggregate",      node_aggregate)
    graph.add_node("write_memory",   node_write_memory)

    # -- Entry point --
    graph.set_entry_point("interpret")

    # -- Edges --

    # interpret -> conditional (ambiguous? end : routing_check)
    graph.add_conditional_edges(
        "interpret",
        should_clarify,
        {"end": END, "routing_check": "routing_check"},
    )

    # routing_check -> conditional (cache hit? execute : plan)
    graph.add_conditional_edges(
        "routing_check",
        cache_hit_or_plan,
        {"execute": "execute", "plan": "plan"},
    )

    # plan -> assign (always)
    graph.add_edge("plan", "assign")

    # assign -> execute (always)
    graph.add_edge("assign", "execute")

    # execute -> conditional (failed+replan? replan : aggregate : end)
    graph.add_conditional_edges(
        "execute",
        after_execute,
        {"replan": "replan", "aggregate": "aggregate", "end": END},
    )

    # replan -> conditional (still failed? end : execute again)
    graph.add_conditional_edges(
        "replan",
        after_replan,
        {"execute": "execute", "end": END},
    )

    # aggregate -> write_memory (always)
    graph.add_edge("aggregate", "write_memory")

    # write_memory -> END
    graph.add_edge("write_memory", END)

    return graph.compile()


# ── Singleton compiled pipeline ────────────────────────────────────────────
# Built once on import, reused for every request.

pipeline = build_pipeline()


# ── Run helper ────────────────────────────────────────────────────────────

def run_pipeline(raw_task: str, session_id: str, user_id: str) -> PipelineState:
    """
    Runs the full pipeline for a task.
    Returns the final PipelineState after all nodes have run.
    """
    initial_state: PipelineState = {
        "raw_task":          raw_task,
        "session_id":        session_id,
        "user_id":           user_id,
        "task_object":       None,
        "cache_hit":         False,
        "few_shot_examples": [],
        "job":               None,
        "agent_tasks":       [],
        "execution_result":  None,
        "final_result":      None,
        "error":             None,
        "replan_count":      0,
        "should_clarify":    False,
        "should_replan":     False,
        "pipeline_done":     False,
    }

    print(f"\n[Pipeline] Starting graph execution")
    final_state = pipeline.invoke(initial_state)
    print(f"[Pipeline] Graph execution complete")

    return final_state
