"""
main.py -- Architect Phase 1 (with LangGraph + Temporal)

The /task endpoint now runs the full LangGraph pipeline.
Everything else (registry, jobs, health) is unchanged.
"""

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from models import TaskRequest, ArchitectResponse, AgentRegisterRequest
from graph.pipeline import run_pipeline
from exporter import read_jobs, read_agent_tasks
from database import init_db
from agent_registry import (
    get_all_agents, get_agent_by_id, register_agent,
    deactivate_agent, get_all_capabilities
)


# -- Startup ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Architect] Starting up...")
    init_db()
    print("[Architect] Ready")
    yield
    print("[Architect] Shutting down")


app = FastAPI(
    title       = "Architect -- Intelligent Orchestrator",
    description = "Phase 1 with LangGraph pipeline + Temporal durable execution",
    version     = "0.3.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# -- Task pipeline ------------------------------------------------------------

@app.post("/task", response_model=ArchitectResponse)
async def submit_task(request: TaskRequest):
    """
    Runs the full LangGraph pipeline for the task.
    Pipeline: interpret -> routing_check -> plan -> assign -> execute (Temporal) -> aggregate -> write_memory
    """
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    user_id    = request.user_id    or "anonymous"

    print(f"\n{'='*60}")
    print(f"[Architect] task='{request.task}'")
    print(f"[Architect] session={session_id} user={user_id}")
    print(f"{'='*60}")

    try:
        final_state = run_pipeline(
            raw_task   = request.task,
            session_id = session_id,
            user_id    = user_id,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Pipeline failed: {e}")

    # -- Ambiguous task --
    if final_state.get("should_clarify"):
        task_obj = final_state.get("task_object")
        return ArchitectResponse(
            success       = False,
            job_id        = "none",
            message       = "Task is ambiguous -- clarification needed",
            job           = _empty_job(request.task, task_obj, session_id, user_id),
            agent_tasks   = [],
            ambiguous     = True,
            clarification = task_obj.ambiguity_note if task_obj else None,
        )

    # -- Pipeline error --
    if final_state.get("error") or not final_state.get("job"):
        raise HTTPException(500, detail=final_state.get("error", "Pipeline failed with no job"))

    job         = final_state["job"]
    agent_tasks = final_state.get("agent_tasks", [])
    final_result = final_state.get("final_result", {})

    return ArchitectResponse(
        success     = True,
        job_id      = job.job_id,
        message     = (
            f"Job complete | {len(job.steps)} step(s) | "
            f"{len(agent_tasks)} agent task(s) | "
            f"status={final_result.get('status', 'complete')}"
        ),
        job         = job,
        agent_tasks = agent_tasks,
    )


# -- Job endpoints ------------------------------------------------------------

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


# -- Registry endpoints -------------------------------------------------------

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


# -- Health -------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "architect", "phase": "1", "version": "0.3.0"}


# -- Helpers ------------------------------------------------------------------

def _empty_job(raw_task, task_object, session_id, user_id):
    from models import Job, JobStatus, Formation
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return Job(
        job_id="none", session_id=session_id, user_id=user_id,
        raw_task=raw_task, task_object=task_object, steps=[],
        status=JobStatus.CREATED, formation=Formation.SEQUENTIAL,
        created_at=now, updated_at=now,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
