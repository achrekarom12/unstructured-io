from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities import greet, start_extraction_job, get_job_status, download_output


@workflow.defn
class SayHelloWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            greet,
            name,
            schedule_to_close_timeout=timedelta(seconds=10),
        )


@workflow.defn
class ExtractionWorkflow:
    """Starts an extraction job, polls until completed, then returns success."""

    @workflow.run
    async def run(
        self,
        input_dir: str,
        job_template_id: str | None = "hi_res_and_enrichment",
        job_nodes: list | None = None,
    ) -> dict:
        # Start the extraction job
        job_info = await workflow.execute_activity(
            start_extraction_job,
            args=[input_dir, job_template_id, job_nodes],
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        job_id = job_info["job_id"]
        job_input_file_ids = job_info["job_input_file_ids"]

        # Poll for completion using workflow.sleep between checks (keeps workflow deterministic)
        poll_interval = timedelta(seconds=10)
        while True:
            status_info = await workflow.execute_activity(
                get_job_status,
                job_id,
                schedule_to_close_timeout=timedelta(minutes=1),
            )
            status = status_info.get("status", "").upper()
            if status == "COMPLETED":
                await workflow.execute_activity(
                    download_output,
                    args=[job_id, job_input_file_ids],
                    schedule_to_close_timeout=timedelta(minutes=1),
                )
                break
            if status not in ("SCHEDULED", "IN_PROGRESS"):
                raise RuntimeError(f"Job ended with status: {status}")
            await workflow.sleep(poll_interval)

        return {
            "success": True,
            "job_id": job_id,
            "job_input_file_ids": job_info.get("job_input_file_ids", []),
            "job_output_node_files": job_info.get("job_output_node_files", []),
        }
