import argparse
import asyncio
import uuid
from pathlib import Path

from temporalio.client import Client

# Project root (parent of temporal-io)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def run_hello():
    """Run the sample SayHello workflow."""
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        "SayHelloWorkflow",
        "Temporal",
        id=f"hello-{uuid.uuid4()}",
        task_queue="sia",
    )
    print("Workflow result:", result)


async def run_extraction(
    input_dir: str | None = None,
    job_template_id: str = "hi_res_and_enrichment",
):
    """Start extraction workflow: start job, poll until complete, return success."""
    if input_dir is None:
        input_dir = str(PROJECT_ROOT / "input")
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        "ExtractionWorkflow",
        args=[input_dir, job_template_id, None],
        id=f"extraction-{uuid.uuid4()}",
        task_queue="sia",
    )
    print("Extraction workflow result:", result)
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Run extraction workflow")
    args = parser.parse_args()
    if args.extract:
        await run_extraction()
    else:
        await run_hello()


if __name__ == "__main__":
    asyncio.run(main())
