"""
models.py -- All Pydantic models used across Architect Phase 1
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Priority(str, Enum):
    FAST     = "fast"
    BALANCED = "balanced"
    CHEAP    = "cheap"
    ACCURATE = "accurate"

class Sensitivity(str, Enum):
    PUBLIC       = "public"
    INTERNAL     = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED   = "restricted"

class StepStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE    = "complete"
    FAILED      = "failed"

class JobStatus(str, Enum):
    CREATED  = "created"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"

class Formation(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL   = "parallel"


class TaskRequest(BaseModel):
    task:       str           = Field(..., description="Raw task string from the user")
    session_id: Optional[str] = Field(None)
    user_id:    Optional[str] = Field(None)


class Constraints(BaseModel):
    priority: Priority = Priority.BALANCED
    cost:     str      = "any"
    accuracy: str      = "high"

class TaskObject(BaseModel):
    goal:                  str
    sub_goals:             List[str]
    required_capabilities: List[str]
    constraints:           Constraints
    sensitivity:           Sensitivity   = Sensitivity.INTERNAL
    is_continuation:       bool          = False
    ambiguous:             bool          = False
    ambiguity_note:        Optional[str] = None
    assumption_notes:      Optional[str] = None
    cot_reasoning:         Optional[str] = None


class JobStep(BaseModel):
    step_id:             str
    action:              str
    required_capability: str
    depends_on:          List[str]     = []
    can_parallel:        bool          = False
    estimated_risk:      str           = "low"
    status:              StepStatus    = StepStatus.PENDING
    assigned_agent_id:   Optional[str] = None

class Job(BaseModel):
    job_id:      str
    session_id:  Optional[str]
    user_id:     Optional[str]
    raw_task:    str
    task_object: TaskObject
    steps:       List[JobStep]
    status:      JobStatus = JobStatus.CREATED
    formation:   Formation = Formation.SEQUENTIAL
    created_at:  str
    updated_at:  str


class AgentTask(BaseModel):
    agent_task_id: str
    job_id:        str
    step_id:       str
    agent_id:      str
    agent_name:    str
    capability:    str
    action:        str
    priority:      Priority
    sensitivity:   Sensitivity
    depends_on:    List[str]      = []
    can_parallel:  bool           = False
    status:        StepStatus     = StepStatus.PENDING
    created_at:    str
    metadata:      Dict[str, Any] = {}


class AgentRegisterRequest(BaseModel):
    agent_id:          str       = Field(..., description="Unique agent ID e.g. agent_my_custom")
    name:              str       = Field(..., description="Human readable name")
    description:       str       = Field(..., description="Plain language description. Used in CoT prompts.")
    capabilities:      List[str] = Field(..., description="List of capability strings")
    speed_score:       int       = Field(3,    ge=1, le=5)
    cost_score:        int       = Field(3,    ge=1, le=5)
    accuracy_score:    float     = Field(0.90, ge=0.0, le=1.0)
    performance_score: float     = Field(0.90, ge=0.0, le=1.0)
    trust_level:       str       = Field("standard")
    version:           str       = Field("1.0.0")


class ArchitectResponse(BaseModel):
    success:       bool
    job_id:        str
    message:       str
    job:           Job
    agent_tasks:   List[AgentTask]
    ambiguous:     bool          = False
    clarification: Optional[str] = None
