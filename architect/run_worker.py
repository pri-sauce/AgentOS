"""
run_worker.py -- Run this from the project root to start the Temporal worker

Usage:
  python run_worker.py
"""

import sys
import os

# Make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
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
    print(f"[Worker] Ready. Keep this running.\n")

    worker = Worker(
        client,
        task_queue = TASK_QUEUE,
        workflows  = [JobWorkflow],
        activities = [execute_agent_task, notify_completion],
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
