"""
exporter.py — JSON Exporter

Appends jobs and agent_tasks to their respective JSON files.
Never overwrites — always appends.

Output files:
  output/jobs.json         — array of all Job objects
  output/agent_tasks.json  — array of all AgentTask objects

File format: a JSON array where each new item is appended.
On first run the file is created. On subsequent runs items are appended.

Thread-safe: uses a file lock to prevent concurrent write corruption.
"""

import json
import os
import threading
from pathlib import Path
from typing import List

from models import Job, AgentTask

# ── Output paths ───────────────────────────────────────────────────────────

OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "output"))
JOBS_FILE        = OUTPUT_DIR / "jobs.json"
AGENT_TASKS_FILE = OUTPUT_DIR / "agent_tasks.json"

# Thread lock — prevents corruption if multiple requests hit simultaneously
_lock = threading.Lock()


def export(job: Job, agent_tasks: List[AgentTask]) -> None:
    """
    Appends the job and its agent tasks to their respective JSON files.
    Creates the output directory and files if they don't exist.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with _lock:
        _append_to_json_file(JOBS_FILE, [job.model_dump()])
        _append_to_json_file(AGENT_TASKS_FILE, [at.model_dump() for at in agent_tasks])

    print(f"[Exporter] Job {job.job_id} written to {JOBS_FILE}")
    print(f"[Exporter] {len(agent_tasks)} agent task(s) written to {AGENT_TASKS_FILE}")


def _append_to_json_file(filepath: Path, new_items: list) -> None:
    """
    Appends new_items to a JSON array file.
    If file doesn't exist → creates it.
    If file exists → loads, appends, writes back.
    """
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            existing = json.loads(content) if content else []
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, OSError):
            existing = []
    else:
        existing = []

    existing.extend(new_items)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)


def read_jobs() -> list:
    """Read all jobs from jobs.json — used by the /jobs endpoint."""
    return _read_json_file(JOBS_FILE)


def read_agent_tasks() -> list:
    """Read all agent tasks from agent_tasks.json — used by /agent-tasks endpoint."""
    return _read_json_file(AGENT_TASKS_FILE)


def _read_json_file(filepath: Path) -> list:
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return []
