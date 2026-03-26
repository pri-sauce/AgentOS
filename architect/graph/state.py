"""
graph/state.py -- LangGraph Pipeline State

v0.4 additions:
  agent_results  — dict of step_id -> agent output, built as steps complete
                   used to pass upstream outputs as inputs to downstream steps
  callback_url   — forwarded from TaskRequest so write_memory can push result back
  replan_reason  — why execution failed, used by LLM-driven replan node
"""

from typing import Optional, List, Any, Dict
from typing_extensions import TypedDict
from models import TaskObject, Job, AgentTask


class PipelineState(TypedDict):
    # -- Input --
    raw_task:     str
    session_id:   str
    user_id:      str
    callback_url: Optional[str]

    # -- After Task Interpreter --
    task_object: Optional[TaskObject]

    # -- After Routing Memory Check --
    cache_hit:         bool
    few_shot_examples: List[dict]

    # -- After Plan / Job Creator --
    job: Optional[Job]

    # -- After Agent Assigner --
    agent_tasks: List[AgentTask]

    # -- Live agent outputs (step_id -> output dict)
    # Each downstream step gets prior step outputs injected as input_data
    agent_results: Dict[str, Any]

    # -- After Execution (Temporal) --
    execution_result: Optional[dict]

    # -- After Result Aggregator --
    final_result: Optional[dict]

    # -- Error / failure state --
    error:         Optional[str]
    replan_count:  int
    replan_reason: Optional[str]

    # -- Pipeline control flags --
    should_clarify: bool
    should_replan:  bool
    pipeline_done:  bool
