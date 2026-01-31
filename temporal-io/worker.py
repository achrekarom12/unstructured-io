import asyncio
import sys
from pathlib import Path

# Ensure project root is on path for main.py imports from activities
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from workflows import SayHelloWorkflow, ExtractionWorkflow
    from activities import greet, start_extraction_job, get_job_status, download_output


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="sia",
        workflows=[SayHelloWorkflow, ExtractionWorkflow],
        activities=[greet, start_extraction_job, get_job_status, download_output],
    )
    print("Worker started.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
