import os
import sys
from pathlib import Path

from temporalio import activity, client

# Ensure project root is on path so we can import main
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from main import run_on_demand_job, download_job_output, get_job_status as get_job_st
from unstructured_client import UnstructuredClient


def _client() -> UnstructuredClient:
    api_key = os.environ.get("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise ValueError("UNSTRUCTURED_API_KEY environment variable is required")
    return UnstructuredClient(api_key_auth=api_key)


@activity.defn
async def greet(name: str) -> str:
    return f"Hello {name}"


@activity.defn
async def start_extraction_job(
    input_dir: str,
    job_template_id: str | None = None,
    job_nodes: list | None = None,
) -> dict:
    """Start an Unstructured extraction job. Returns job_id and input file ids."""
    with _client() as client:
        job_id, job_input_file_ids, job_output_node_files = run_on_demand_job(
            client=client,
            input_dir=input_dir,
            job_template_id=job_template_id,
            job_nodes=job_nodes,
        )
    return {
        "job_id": job_id,
        "job_input_file_ids": job_input_file_ids,
        "job_output_node_files": job_output_node_files,
    }

@activity.defn
async def download_output(
    job_id: str,
    job_input_file_ids: list[str],
) -> None:
    with _client() as client:
        download_job_output(
            client=client,
            output_dir="../output",
            job_id=job_id,
            job_input_file_ids=job_input_file_ids,
        )


@activity.defn
async def get_job_status(job_id: str) -> dict:
    """Fetch current job status (one API call). Returns serializable job info with 'status'."""
    with _client() as client:
        return get_job_st(client, job_id)
