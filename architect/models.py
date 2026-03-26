"""
models.py -- All Pydantic models used across Architect

v0.4 additions:
  HeartbeatRequest    — agent pings /registry/agents/{id}/heartbeat
  AgentDiscoveryQuery — structured query for /registry/agents/discover
  AgentHealth         — health status response per agent
  AgentOutputPacket   — what an agent returns after executing a task
  StepInput           — what Architect sends to an agent (instruction packet)
  JobResultRecord     — persisted job result stored in PostgreSQL
  CallbackPayload     — pushed to Interaction Adapter when a job finishes
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

class CertificationStatus(str, Enum):
    UNCERTIFIED = "uncertified"
    STANDARD    = "standard"
    VERIFIED    = "verified"
    RESTRICTED  = "restricted"

class HealthStatus(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    OFFLINE  = "offline"
    UNKNOWN  = "unknown"


# ── Incoming request ───────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task:         str           = Field(..., description="Raw task string from the user")
    session_id:   Optional[str] = Field(None)
    user_id:      Optional[str] = Field(None)
    callback_url: Optional[str] = Field(None, description="URL to POST result to when job completes")


# ── Task interpretation ────────────────────────────────────────────────────

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


# ── Job + steps ────────────────────────────────────────────────────────────

class JobStep(BaseModel):
    step_id:             str
    action:              str
    required_capability: str
    depends_on:          List[str]  = []
    can_parallel:        bool       = False
    estimated_risk:      str        = "low"
    status:              StepStatus = StepStatus.PENDING
    assigned_agent_id:   Optional[str] = None
    input_from_steps:    List[str]  = Field(
        default=[],
        description="step_ids whose output should be passed as input to this step"
    )

class Job(BaseModel):
    job_id:       str
    session_id:   Optional[str]
    user_id:      Optional[str]
    raw_task:     str
    task_object:  TaskObject
    steps:        List[JobStep]
    status:       JobStatus = JobStatus.CREATED
    formation:    Formation = Formation.SEQUENTIAL
    created_at:   str
    updated_at:   str
    callback_url: Optional[str] = None


# ── Agent task (dispatch unit) ────────────────────────────────────────────

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
    input_data:    Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output from upstream steps passed as input to this agent"
    )


# ── What Architect sends to Agent Connector ───────────────────────────────

class StepInput(BaseModel):
    """Instruction packet sent to Agent Connector for each task."""
    job_id:        str
    step_id:       str
    agent_task_id: str
    action:        str
    capability:    str
    sensitivity:   str
    priority:      str
    input_data:    Dict[str, Any] = Field(default_factory=dict)
    metadata:      Dict[str, Any] = {}


# ── What Agent Connector returns ──────────────────────────────────────────

class AgentOutputPacket(BaseModel):
    """
    Standardised response from Agent Connector.
    Flows back into PipelineState.agent_results[step_id].
    """
    step_id:      str
    agent_id:     str
    agent_name:   str
    action:       str
    status:       StepStatus
    output:       Dict[str, Any] = Field(default_factory=dict)
    error:        Optional[str]  = None
    started_at:   str
    completed_at: str
    duration_ms:  int


# ── Agent registry management ─────────────────────────────────────────────

class AgentRegisterRequest(BaseModel):
    agent_id:             str       = Field(..., description="Unique agent ID")
    name:                 str
    description:          str
    capabilities:         List[str]
    speed_score:          int       = Field(3,    ge=1, le=5)
    cost_score:           int       = Field(3,    ge=1, le=5)
    accuracy_score:       float     = Field(0.90, ge=0.0, le=1.0)
    performance_score:    float     = Field(0.90, ge=0.0, le=1.0)
    trust_level:          str       = Field("standard")
    certification_status: str       = Field("uncertified")
    version:              str       = Field("1.0.0")
    endpoint_url:         Optional[str] = Field(None)


# ── Heartbeat ─────────────────────────────────────────────────────────────

class HeartbeatRequest(BaseModel):
    agent_id: str
    status:   HealthStatus = HealthStatus.HEALTHY
    version:  Optional[str]            = None
    metadata: Optional[Dict[str, Any]] = None

class HeartbeatResponse(BaseModel):
    agent_id:    str
    received_at: str
    status:      str = "ok"


# ── Agent health ──────────────────────────────────────────────────────────

class AgentHealth(BaseModel):
    agent_id:             str
    name:                 str
    is_active:            bool
    health_status:        HealthStatus
    last_seen:            Optional[str]
    seconds_since_seen:   Optional[float]
    certification_status: str
    trust_level:          str
    endpoint_url:         Optional[str]


# ── Agent discovery ───────────────────────────────────────────────────────

class AgentDiscoveryQuery(BaseModel):
    capability:        Optional[str]  = None
    min_trust_level:   Optional[str]  = None
    min_certification: Optional[str]  = None
    priority:          Optional[str]  = "balanced"
    exclude_agents:    List[str]      = []
    require_endpoint:  bool           = False
    healthy_only:      bool           = True
    limit:             int            = Field(5, ge=1, le=20)


# ── Job result persistence ────────────────────────────────────────────────

class JobResultRecord(BaseModel):
    job_id:        str
    session_id:    Optional[str]
    user_id:       Optional[str]
    raw_task:      str
    goal:          Optional[str]
    status:        str
    steps_run:     List[Dict[str, Any]] = []
    agents_used:   List[str]            = []
    final_output:  Dict[str, Any]       = {}
    total_time_ms: int                  = 0
    completed_at:  str


# ── Outbound callback ─────────────────────────────────────────────────────

class CallbackPayload(BaseModel):
    """Posted to callback_url when a job finishes."""
    job_id:        str
    session_id:    Optional[str]
    status:        str
    summary:       str
    final_output:  Dict[str, Any] = {}
    agents_used:   List[str]      = []
    total_time_ms: int            = 0
    completed_at:  str


# ── API responses ─────────────────────────────────────────────────────────

class ArchitectResponse(BaseModel):
    success:       bool
    job_id:        str
    message:       str
    job:           Job
    agent_tasks:   List[AgentTask]
    final_result:  Optional[Dict[str, Any]] = None
    ambiguous:     bool          = False
    clarification: Optional[str] = None
