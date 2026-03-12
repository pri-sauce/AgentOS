"""
temporal/worker.py -- Temporal Worker

Run this as a SEPARATE process alongside the FastAPI server.
The worker listens on the "architect-task-queue" and executes
activities when the workflow dispatches them.

Usage:
  python temporal/worker.py

Keep this running while the FastAPI server is running.
Both need to be running for tasks to execute fully.

What happens if the worker crashes:
  Temporal holds the pending activities.
  When the worker restarts, it picks up from where it left off.
  No tasks are lost.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from temporalio.client import Client
from temporalio.worker import Worker

from temporal.workflows import JobWorkflow
from temporal.activities import execute_agent_task, notify_completion

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TASK_QUEUE    = "architect-task-queue"


async def main():
    print(f"[Worker] Connecting to Temporal at {TEMPORAL_HOST}...")
    client = await Client.connect(TEMPORAL_HOST)
    print(f"[Worker] Connected. Listening on queue: {TASK_QUEUE}")
    print(f"[Worker] Ready to process jobs. Keep this running.")
    print(f"[Worker] Press Ctrl+C to stop.\n")

    worker = Worker(
        client,
        task_queue  = TASK_QUEUE,
        workflows   = [JobWorkflow],
        activities  = [execute_agent_task, notify_completion],
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
