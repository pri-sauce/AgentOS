"""
agent_assigner.py -- Agent Assigner

For each step in a job, picks the best agent from the PostgreSQL registry.
Selection factors: required capability + priority constraint + performance score.

Phase 1: direct registry query
Later phases: pgvector semantic shortlist -> LLM final pick with CoT
"""

import uuid
from datetime import datetime, timezone
from typing import List

from models import Job, AgentTask, StepStatus
from agent_registry import pick_best_agent


def assign_agents(job: Job) -> List[AgentTask]:
    """
    Produces one AgentTask per step, assigning the best agent from the registry.
    """
    agent_tasks: List[AgentTask] = []

    for step in job.steps:
        agent = pick_best_agent(
            capability = step.required_capability,
            priority   = job.task_object.constraints.priority.value,
        )

        agent_task = AgentTask(
            agent_task_id = f"at_{uuid.uuid4().hex[:12]}",
            job_id        = job.job_id,
            step_id       = step.step_id,
            agent_id      = agent["agent_id"],
            agent_name    = agent["name"],
            capability    = step.required_capability,
            action        = step.action,
            priority      = job.task_object.constraints.priority,
            sensitivity   = job.task_object.sensitivity,
            depends_on    = step.depends_on,
            can_parallel  = step.can_parallel,
            status        = StepStatus.PENDING,
            created_at    = _now(),
            metadata      = {
                "job_goal":          job.task_object.goal,
                "estimated_risk":    step.estimated_risk,
                "agent_speed":       agent.get("speed_score"),
                "agent_cost":        agent.get("cost_score"),
                "agent_accuracy":    agent.get("accuracy_score"),
                "agent_performance": agent.get("performance_score"),
                "agent_trust":       agent.get("trust_level"),
                "selection_basis":   f"priority={job.task_object.constraints.priority.value}",
            }
        )

        step.assigned_agent_id = agent["agent_id"]
        agent_tasks.append(agent_task)

        print(
            f"[AgentAssigner] {step.step_id} -> {agent['name']} ({agent['agent_id']}) "
            f"| perf={agent.get('performance_score', 0):.0%} speed={agent.get('speed_score')}"
        )

    return agent_tasks


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
