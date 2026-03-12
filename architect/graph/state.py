"""
graph/state.py -- LangGraph Pipeline State

This is the single state object that flows through every node in the graph.
Each node reads from it and writes back to it.
LangGraph passes this state between nodes automatically.

Think of it as the shared memory of one pipeline run.
"""

from typing import Optional, List, Any
from typing_extensions import TypedDict
from models import TaskObject, Job, AgentTask


class PipelineState(TypedDict):
    """
    Shared state across all LangGraph nodes.
    Every field starts as None and gets filled as nodes run.
    """

    # -- Input --
    raw_task:    str
    session_id:  str
    user_id:     str

    # -- After Task Interpreter --
    task_object: Optional[TaskObject]

    # -- After Routing Memory Check --
    cache_hit:        bool           # True if routing memory had a confident match
    few_shot_examples: List[dict]    # pulled from routing memory regardless of hit/miss

    # -- After Plan Generator / Job Creator --
    job: Optional[Job]

    # -- After Agent Assigner --
    agent_tasks: List[AgentTask]

    # -- After Execution (Temporal) --
    execution_result: Optional[dict]  # temporal workflow result

    # -- After Result Aggregator --
    final_result: Optional[dict]

    # -- Error / failure state --
    error:        Optional[str]   # set if any node fails
    replan_count: int             # how many times we've replanned

    # -- Pipeline control flags --
    should_clarify:  bool   # True if task was ambiguous -> return early
    should_replan:   bool   # True if execution failed and replan is needed
    pipeline_done:   bool   # True when pipeline is complete
