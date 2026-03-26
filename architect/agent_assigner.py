"""
agent_assigner.py -- Agent Assigner

For each step in a job, picks the best agent from the PostgreSQL registry.
v0.4: now passes sensitivity to pick_best_agent() for trust gate enforcement.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from models import Job, AgentTask, StepStatus
from agent_registry import pick_best_agent


def assign_agents(job: Job) -> List[AgentTask]:
    """
    Produces one AgentTask per step, assigning the best agent from the registry.
    Trust gate is enforced based on job sensitivity.
    """
    agent_tasks: List[AgentTask] = []
    sensitivity = job.task_object.sensitivity.value

    for step in job.steps:
        agent = pick_best_agent(
            capability  = step.required_capability,
            priority    = job.task_object.constraints.priority.value,
            sensitivity = sensitivity,
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
                "agent_cert":        agent.get("certification_status"),
                "endpoint_url":      agent.get("endpoint_url"),
                "selection_basis":   f"priority={job.task_object.constraints.priority.value} sensitivity={sensitivity}",
                "input_from_steps":  step.input_from_steps,
            }
        )

        step.assigned_agent_id = agent["agent_id"]
        agent_tasks.append(agent_task)

        print(
            f"[AgentAssigner] {step.step_id} -> {agent['name']} ({agent['agent_id']}) "
            f"| perf={agent.get('performance_score', 0):.0%} trust={agent.get('trust_level')} "
            f"cert={agent.get('certification_status')}"
        )

    return agent_tasks


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
