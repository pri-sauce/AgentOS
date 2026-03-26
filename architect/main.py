"""
main.py -- Architect v0.4

New endpoints:
  POST /registry/agents/{agent_id}/heartbeat   — agent pings in, updates last_seen
  GET  /registry/agents/{agent_id}/health      — health status for one agent
  GET  /registry/health                        — health overview for all agents
  POST /registry/agents/discover               — structured discovery query

New behaviours:
  - Background task runs every 60s to mark stale agents offline
  - callback_url forwarded from TaskRequest through pipeline
  - /task now returns final_result in response
"""

import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from models import (
    TaskRequest, ArchitectResponse, AgentRegisterRequest,
    HeartbeatRequest, HeartbeatResponse, AgentDiscoveryQuery,
)
from graph.pipeline import run_pipeline
from exporter import read_jobs, read_agent_tasks
from database import init_db
from agent_registry import (
    get_all_agents, get_agent_by_id, register_agent,
    deactivate_agent, get_all_capabilities,
    update_heartbeat, mark_stale_agents,
    get_agent_health, get_all_health, discover_agents,
)


# ── Background health monitor ─────────────────────────────────────────────

async def _health_monitor_loop():
    """
    Runs every 60 seconds.
    Marks agents as inactive if they haven't sent a heartbeat in > 5 minutes.
    """
    while True:
        await asyncio.sleep(60)
        try:
            count = mark_stale_agents(stale_after_seconds=300)
            if count > 0:
                print(f"[HealthMonitor] Marked {count} agent(s) offline")
        except Exception as e:
            print(f"[HealthMonitor] Error: {e}")


# ── Startup / shutdown ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Architect] Starting up...")
    init_db()
    # Start background health monitor
    monitor_task = asyncio.create_task(_health_monitor_loop())
    print("[Architect] Health monitor started (60s interval)")
    print("[Architect] Ready on v0.4")
    yield
    monitor_task.cancel()
    print("[Architect] Shutting down")


app = FastAPI(
    title       = "Architect — Intelligent Orchestrator",
    description = "v0.4: LangGraph + Temporal + Agent Health + Output Chaining",
    version     = "0.4.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Task pipeline ─────────────────────────────────────────────────────────

@app.post("/task", response_model=ArchitectResponse)
async def submit_task(request: TaskRequest):
    """
    Full LangGraph pipeline:
    interpret -> routing_check -> plan -> assign -> execute (Temporal) -> aggregate -> write_memory

    callback_url: if provided, Architect POSTs result to this URL when job finishes.
    """
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    user_id    = request.user_id    or "anonymous"

    print(f"\n{'='*60}")
    print(f"[Architect] task='{request.task}'")
    print(f"[Architect] session={session_id} user={user_id}")
    if request.callback_url:
        print(f"[Architect] callback_url={request.callback_url}")
    print(f"{'='*60}")

    try:
        final_state = run_pipeline(
            raw_task     = request.task,
            session_id   = session_id,
            user_id      = user_id,
            callback_url = request.callback_url,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Pipeline failed: {e}")

    # Ambiguous task
    if final_state.get("should_clarify"):
        task_obj = final_state.get("task_object")
        return ArchitectResponse(
            success       = False,
            job_id        = "none",
            message       = "Task is ambiguous — clarification needed",
            job           = _empty_job(request.task, task_obj, session_id, user_id),
            agent_tasks   = [],
            ambiguous     = True,
            clarification = task_obj.ambiguity_note if task_obj else None,
        )

    # Pipeline error
    if final_state.get("error") or not final_state.get("job"):
        raise HTTPException(500, detail=final_state.get("error", "Pipeline failed with no job"))

    job          = final_state["job"]
    agent_tasks  = final_state.get("agent_tasks", [])
    final_result = final_state.get("final_result", {})

    step_count  = len(job.steps)
    agent_count = len(agent_tasks)
    status      = final_result.get("status", "complete")
    time_ms     = final_result.get("total_time_ms", 0)

    return ArchitectResponse(
        success      = True,
        job_id       = job.job_id,
        message      = f"Job {status} | {step_count} step(s) | {agent_count} agent(s) | {time_ms}ms",
        job          = job,
        agent_tasks  = agent_tasks,
        final_result = final_result,
    )


# ── Job endpoints ─────────────────────────────────────────────────────────

@app.get("/jobs")
async def get_jobs():
    jobs = read_jobs()
    return {"jobs": jobs, "count": len(jobs)}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    for job in read_jobs():
        if job.get("job_id") == job_id:
            return job
    raise HTTPException(404, detail=f"Job {job_id} not found")

@app.get("/agent-tasks")
async def get_agent_tasks():
    tasks = read_agent_tasks()
    return {"agent_tasks": tasks, "count": len(tasks)}

@app.get("/agent-tasks/{job_id}")
async def get_agent_tasks_for_job(job_id: str):
    tasks = [t for t in read_agent_tasks() if t.get("job_id") == job_id]
    if not tasks:
        raise HTTPException(404, detail=f"No agent tasks for job {job_id}")
    return {"job_id": job_id, "agent_tasks": tasks, "count": len(tasks)}


# ── Registry — CRUD ───────────────────────────────────────────────────────

@app.get("/registry/agents")
async def list_agents(active_only: bool = True):
    agents = get_all_agents(active_only=active_only)
    return {"agents": agents, "count": len(agents)}

@app.get("/registry/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(404, detail=f"Agent {agent_id} not found")
    return agent

@app.post("/registry/agents", status_code=201)
async def add_agent(request: AgentRegisterRequest):
    try:
        agent = register_agent(request.model_dump())
        return {"success": True, "message": f"Agent '{request.agent_id}' registered", "agent": dict(agent)}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to register agent: {e}")

@app.delete("/registry/agents/{agent_id}")
async def remove_agent(agent_id: str):
    success = deactivate_agent(agent_id)
    if not success:
        raise HTTPException(404, detail=f"Agent {agent_id} not found")
    return {"success": True, "message": f"Agent {agent_id} deactivated"}

@app.get("/registry/capabilities")
async def list_capabilities():
    caps = get_all_capabilities()
    return {"capabilities": caps, "count": len(caps)}


# ── Registry — Heartbeat ──────────────────────────────────────────────────

@app.post("/registry/agents/{agent_id}/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(agent_id: str, request: HeartbeatRequest):
    """
    Called by agents (or Agent Connector on their behalf) every N seconds.
    Updates last_seen. If agent was marked inactive due to staleness, reactivates it.
    """
    if request.agent_id != agent_id:
        raise HTTPException(400, detail="agent_id in body must match URL")

    result = update_heartbeat(
        agent_id = agent_id,
        status   = request.status.value,
        version  = request.version,
    )

    if not result:
        raise HTTPException(404, detail=f"Agent {agent_id} not found in registry")

    return HeartbeatResponse(
        agent_id    = agent_id,
        received_at = datetime.now(timezone.utc).isoformat(),
        status      = "ok",
    )


# ── Registry — Health ─────────────────────────────────────────────────────

@app.get("/registry/agents/{agent_id}/health")
async def agent_health(agent_id: str):
    """
    Returns health status for a single agent.
    health_status: healthy | degraded | offline | unknown
    """
    health = get_agent_health(agent_id)
    if not health:
        raise HTTPException(404, detail=f"Agent {agent_id} not found")
    return health

@app.get("/registry/health")
async def all_agent_health():
    """
    Health overview for all agents.
    Shows last_seen, health_status, certification_status per agent.
    """
    health_list = get_all_health()
    summary = {
        "healthy":  sum(1 for h in health_list if h["health_status"] == "healthy"),
        "degraded": sum(1 for h in health_list if h["health_status"] == "degraded"),
        "offline":  sum(1 for h in health_list if h["health_status"] == "offline"),
        "unknown":  sum(1 for h in health_list if h["health_status"] == "unknown"),
    }
    return {
        "agents": health_list,
        "count":  len(health_list),
        "summary": summary,
    }


# ── Registry — Discovery ──────────────────────────────────────────────────

@app.post("/registry/agents/discover")
async def discover(query: AgentDiscoveryQuery):
    """
    Structured agent discovery used by the Workflow Manager and external callers.

    Filters by: capability, trust_level, certification_status, health, endpoint availability.
    Orders by priority: fast | cheap | accurate | balanced.

    Example:
      POST /registry/agents/discover
      {"capability": "summarisation", "min_trust_level": "verified", "healthy_only": true}
    """
    agents = discover_agents(
        capability        = query.capability,
        min_trust_level   = query.min_trust_level,
        min_certification = query.min_certification,
        priority          = query.priority or "balanced",
        exclude_agents    = query.exclude_agents,
        require_endpoint  = query.require_endpoint,
        healthy_only      = query.healthy_only,
        limit             = query.limit,
    )
    return {
        "agents": agents,
        "count":  len(agents),
        "query":  query.model_dump(),
    }


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "architect",
        "version": "0.4.0",
        "phase":   "1+",
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _empty_job(raw_task, task_object, session_id, user_id):
    from models import Job, JobStatus, Formation
    now = datetime.now(timezone.utc).isoformat()
    return Job(
        job_id="none", session_id=session_id, user_id=user_id,
        raw_task=raw_task, task_object=task_object, steps=[],
        status=JobStatus.CREATED, formation=Formation.SEQUENTIAL,
        created_at=now, updated_at=now,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9999, reload=True)
