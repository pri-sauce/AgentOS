"""
main.py -- Architect v0.5

v0.5 additions:
  - Consul integration: on startup, syncs all PostgreSQL agents to Consul
  - Background health monitor: every 60s marks stale agents + syncs Consul → PostgreSQL
  - Heartbeat endpoint now updates Consul TTL check as well as PostgreSQL last_seen
  - /registry/health and /registry/agents/{id}/health now enriched with Consul status
  - POST /registry/agents now registers in both PostgreSQL and Consul
  - DELETE /registry/agents now deregisters from both
  - GET /registry/consul — Consul service catalog view
"""

import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
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
    get_all_agents, get_agent_by_id,
    deactivate_agent, get_all_capabilities,
    update_heartbeat, mark_stale_agents,
    get_agent_health, get_all_health, discover_agents,
    # Consul-enriched versions
    get_agent_health_enriched, get_all_health_enriched,
    register_agent_with_consul, deactivate_agent_with_consul,
)
from consul_registry import (
    consul_available, sync_postgres_to_consul,
    sync_consul_to_postgres, update_consul_ttl,
    get_all_consul_services,
)


# ── Background health monitor ─────────────────────────────────────────────

async def _health_monitor_loop():
    """
    Runs every 60 seconds:
    1. Marks agents offline in PostgreSQL if no heartbeat for > 5 min
    2. Syncs Consul health results back into PostgreSQL is_active
    """
    while True:
        await asyncio.sleep(60)
        try:
            stale = mark_stale_agents(stale_after_seconds=300)
            if stale:
                print(f"[HealthMonitor] Marked {stale} agent(s) offline (no heartbeat)")
        except Exception as e:
            print(f"[HealthMonitor] Stale check error: {e}")

        try:
            changed = sync_consul_to_postgres()
            if changed:
                print(f"[HealthMonitor] Consul sync updated {changed} agent(s) in PostgreSQL")
        except Exception as e:
            print(f"[HealthMonitor] Consul sync error: {e}")


# ── Startup / shutdown ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Architect] Starting up...")
    init_db()

    # Register all active PostgreSQL agents in Consul
    if consul_available():
        print("[Architect] Consul is available — syncing agents...")
        synced = sync_postgres_to_consul()
        print(f"[Architect] {synced} agent(s) registered in Consul")
    else:
        print("[Architect] Consul not available — health checking via heartbeat only")

    monitor_task = asyncio.create_task(_health_monitor_loop())
    print("[Architect] Health monitor started")
    print("[Architect] Ready on v0.5")
    yield
    monitor_task.cancel()
    print("[Architect] Shutting down")


app = FastAPI(
    title       = "Architect — Intelligent Orchestrator",
    description = "v0.5: LangGraph + Temporal + Consul Service Discovery",
    version     = "0.5.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Task pipeline ─────────────────────────────────────────────────────────

@app.post("/task", response_model=ArchitectResponse)
async def submit_task(request: TaskRequest):
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

    if final_state.get("error") or not final_state.get("job"):
        raise HTTPException(500, detail=final_state.get("error", "Pipeline failed"))

    job          = final_state["job"]
    agent_tasks  = final_state.get("agent_tasks", [])
    final_result = final_state.get("final_result", {})

    return ArchitectResponse(
        success      = True,
        job_id       = job.job_id,
        message      = (
            f"Job {final_result.get('status', 'complete')} | "
            f"{len(job.steps)} step(s) | {len(agent_tasks)} agent(s) | "
            f"{final_result.get('total_time_ms', 0)}ms"
        ),
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


# ── Registry — CRUD (now Consul-aware) ───────────────────────────────────

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
    """Registers agent in PostgreSQL + Consul."""
    try:
        agent = register_agent_with_consul(request.model_dump())
        return {
            "success": True,
            "message": f"Agent '{request.agent_id}' registered",
            "agent":   dict(agent),
            "consul":  consul_available(),
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to register agent: {e}")

@app.delete("/registry/agents/{agent_id}")
async def remove_agent(agent_id: str):
    """Deactivates in PostgreSQL + deregisters from Consul."""
    success = deactivate_agent_with_consul(agent_id)
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
    Agent pings in.
    Updates PostgreSQL last_seen AND Consul TTL check simultaneously.
    For agents with endpoint_url: Consul handles health checking directly,
    this endpoint still updates last_seen for the PostgreSQL-side record.
    """
    if request.agent_id != agent_id:
        raise HTTPException(400, detail="agent_id in body must match URL")

    # Update PostgreSQL
    result = update_heartbeat(
        agent_id = agent_id,
        status   = request.status.value,
        version  = request.version,
    )
    if not result:
        raise HTTPException(404, detail=f"Agent {agent_id} not found in registry")

    # Update Consul TTL (if agent uses TTL check — no-op for HTTP-checked agents)
    healthy = request.status.value in ("healthy", "degraded")
    update_consul_ttl(agent_id, healthy=healthy)

    return HeartbeatResponse(
        agent_id    = agent_id,
        received_at = datetime.now(timezone.utc).isoformat(),
        status      = "ok",
    )


# ── Registry — Health (Consul-enriched) ───────────────────────────────────

@app.get("/registry/agents/{agent_id}/health")
async def agent_health(agent_id: str):
    """
    Health status enriched with Consul data when available.
    health_source: "consul" | "heartbeat"
    consul_status: "passing" | "warning" | "critical" | null
    """
    health = get_agent_health_enriched(agent_id)
    if not health:
        raise HTTPException(404, detail=f"Agent {agent_id} not found")
    return health

@app.get("/registry/health")
async def all_agent_health():
    """
    Health overview for all agents.
    Enriched with Consul active health check results when Consul is available.
    """
    health_list = get_all_health_enriched()
    summary = {
        "healthy":  sum(1 for h in health_list if h["health_status"] == "healthy"),
        "degraded": sum(1 for h in health_list if h["health_status"] == "degraded"),
        "offline":  sum(1 for h in health_list if h["health_status"] == "offline"),
        "unknown":  sum(1 for h in health_list if h["health_status"] == "unknown"),
    }
    return {
        "agents":         health_list,
        "count":          len(health_list),
        "summary":        summary,
        "consul_active":  consul_available(),
    }


# ── Registry — Consul catalog ─────────────────────────────────────────────

@app.get("/registry/consul")
async def consul_catalog():
    """
    Returns the Consul service catalog for all registered architect-agents.
    Shows active HTTP/TTL check results directly from Consul.
    Returns empty list gracefully if Consul is not running.
    """
    if not consul_available():
        return {
            "available": False,
            "message":   "Consul is not running — start with docker-compose up",
            "services":  [],
        }
    services = get_all_consul_services()
    return {
        "available": True,
        "services":  services,
        "count":     len(services),
        "ui":        "http://localhost:8500",
    }


# ── Registry — Discovery ──────────────────────────────────────────────────

@app.post("/registry/agents/discover")
async def discover(query: AgentDiscoveryQuery):
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


# ── System health ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "service":        "architect",
        "version":        "0.5.0",
        "consul":         consul_available(),
        "consul_ui":      "http://localhost:8500" if consul_available() else None,
        "temporal_ui":    "http://localhost:8080",
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
