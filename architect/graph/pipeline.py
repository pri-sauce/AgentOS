"""
graph/pipeline.py -- LangGraph Pipeline Assembly

v0.4: run_pipeline now accepts callback_url and agent_results initial state.
"""

from langgraph.graph import StateGraph, END

from graph.state import PipelineState
from graph.nodes import (
    node_interpret, node_routing_check, node_plan, node_assign,
    node_execute, node_replan, node_aggregate, node_write_memory,
    should_clarify, cache_hit_or_plan, after_execute, after_replan,
)


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("interpret",    node_interpret)
    graph.add_node("routing_check", node_routing_check)
    graph.add_node("plan",         node_plan)
    graph.add_node("assign",       node_assign)
    graph.add_node("execute",      node_execute)
    graph.add_node("replan",       node_replan)
    graph.add_node("aggregate",    node_aggregate)
    graph.add_node("write_memory", node_write_memory)

    graph.set_entry_point("interpret")

    graph.add_conditional_edges("interpret",     should_clarify,    {"end": END, "routing_check": "routing_check"})
    graph.add_conditional_edges("routing_check", cache_hit_or_plan, {"execute": "execute", "plan": "plan"})
    graph.add_edge("plan",    "assign")
    graph.add_edge("assign",  "execute")
    graph.add_conditional_edges("execute", after_execute, {"replan": "replan", "aggregate": "aggregate", "end": END})
    graph.add_conditional_edges("replan",  after_replan,  {"execute": "execute", "end": END})
    graph.add_edge("aggregate",    "write_memory")
    graph.add_edge("write_memory", END)

    return graph.compile()


pipeline = build_pipeline()


def run_pipeline(
    raw_task:     str,
    session_id:   str,
    user_id:      str,
    callback_url: str = None,
) -> PipelineState:
    """Runs the full pipeline. Returns final PipelineState."""
    initial_state: PipelineState = {
        "raw_task":          raw_task,
        "session_id":        session_id,
        "user_id":           user_id,
        "callback_url":      callback_url,
        "task_object":       None,
        "cache_hit":         False,
        "few_shot_examples": [],
        "job":               None,
        "agent_tasks":       [],
        "agent_results":     {},
        "execution_result":  None,
        "final_result":      None,
        "error":             None,
        "replan_count":      0,
        "replan_reason":     None,
        "should_clarify":    False,
        "should_replan":     False,
        "pipeline_done":     False,
    }

    print(f"\n[Pipeline] Starting graph execution")
    final_state = pipeline.invoke(initial_state)
    print(f"[Pipeline] Graph execution complete")
    return final_state
