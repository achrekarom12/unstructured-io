import argparse
import json
import os
import time
from typing import Optional
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models.operations import CreateJobRequest
from unstructured_client.models.operations import DownloadJobOutputRequest
from unstructured_client.models.shared import BodyCreateJob, InputFiles, JobInformation

load_dotenv()


class UnstructuredIO:
    def __init__(self,
        api_key: str,
    ):
        self.api_key = api_key
        self.client = UnstructuredClient(api_key_auth=self.api_key)

    def run_on_demand_job(
            self,
            input_file: str,
            job_template_id: Optional[str] = None, 
            job_nodes: Optional[list[dict[str, object]]] = None
    ) -> tuple[str, list[dict[str, str]]]:
        """Runs an Unstructured on-demand job."""
        if not os.path.isfile(input_file):
            raise ValueError(f"Input path is not a file or does not exist: {input_file}")

        request_data = {}
        filename = os.path.basename(input_file)
        files = [
            (
                InputFiles(
                    content=open(input_file, "rb"),
                    file_name=filename,
                    content_type="application/pdf"
                )
            )
        ]

        if job_template_id is not None:
            request_data = json.dumps({"template_id": job_template_id})
        elif job_nodes is not None:
            request_data = json.dumps({"job_nodes": job_nodes})
        else:
            raise ValueError(f"Must specify a job template ID or job nodes (but not both).")
            exit(1)

        response = self.client.jobs.create_job(
            request=CreateJobRequest(
                body_create_job=BodyCreateJob(
                    request_data=request_data,
                    input_files=files
                )
            )
        )

        job_id = response.job_information.id
        job_input_file_ids = response.job_information.input_file_ids
        job_output_node_files = response.job_information.output_node_files

        return job_id, job_input_file_ids, job_output_node_files


    def get_job_status(self, job_id: str) -> dict:
        """Returns the current status of a job (one API call)."""
        response = self.client.jobs.get_job(request={"job_id": job_id})
        job = response.job_information
        return job.model_dump(mode="json")


    def poll_for_job_status(self, job_id: str) -> JobInformation:
        """Keeps checking a job's status until the job is completed."""
        while True:
            response = self.client.jobs.get_job(
                request={
                    "job_id": job_id
                }
            )

            job = response.job_information

            if job.status == "SCHEDULED":
                print("Job is scheduled, polling again in 10 seconds...")
                time.sleep(10)
            elif job.status == "IN_PROGRESS":
                print("Job is in progress, polling again in 10 seconds...")
                time.sleep(10)
            else:
                print("Job is completed.")
                break

        return job


    def download_job_output(
            self,
            job_id: str,
            job_input_file_ids: list[str],
            output_dir: str
    ) -> None:
        """Downloads the output of an Unstructured job"""
        for job_input_file_id in job_input_file_ids:
            print(f"Attempting to get processed results from file_id '{job_input_file_id}'...")

            response = self.client.jobs.download_job_output(
                request=DownloadJobOutputRequest(
                    job_id=job_id,
                    file_id=job_input_file_id
                )
            )

            # print(response.any) will give final JSON 

            output_path = os.path.join(output_dir, f"{job_input_file_id}.json")
            print(f"Output path: {output_path}")

            with open(output_path, "w") as f:
                json.dump(response.any, f, indent=4)

            print(f"Saved output for file_id '{job_input_file_id}' to '{output_path}'.\n")


def main():
    parser = argparse.ArgumentParser(description="Run an Unstructured on-demand job on a single file.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input file to process (e.g. --file=./input/document.pdf)",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Directory to save the job output (default: ./output)",
    )
    args = parser.parse_args()

    # API key and paths.
    UNSTRUCTURED_API_KEY = os.environ.get('UNSTRUCTURED_API_KEY')
    INPUT_FILE_PATH = os.path.abspath(args.file)
    OUTPUT_FOLDER_PATH = args.output

    # On-demand job settings.
    job_template_id = "hi_res_and_enrichment"
    job_nodes = [] # Applies only if the job is to use a custom workflow definition.

    # Internal tracking variables.
    job_id = ""
    job_input_file_ids = []
    job_output_node_files = []

    unstructured = UnstructuredIO(
        api_key=UNSTRUCTURED_API_KEY
    )

    print("-" * 80)
    print(f"Attempting to run the on-demand job on input file '{INPUT_FILE_PATH}'...")
    job_id, job_input_file_ids, job_output_node_files = unstructured.run_on_demand_job(
        input_file=INPUT_FILE_PATH,
        job_template_id=job_template_id
    )

    print(f"Job ID: {job_id}\n")
    print("Input file details:\n")

    for job_input_file_id in job_input_file_ids:
        print(job_input_file_id)

    print("\nOutput node file details:\n")

    for output_node_file in job_output_node_files:
        print(output_node_file)

    print("-" * 80)
    print("Polling for job status...")

    job = unstructured.poll_for_job_status(job_id=job_id)
    
    print(f"Job details:\n---\n{job.model_dump_json(indent=4)}")

    if job.status != "COMPLETED":
        print("Job did not complete successfully. Stopping this script without downloading any output.")
        exit(1)

    print("-" * 80)
    print("Attempting to download the job output...")
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    unstructured.download_job_output(job_id=job_id, job_input_file_ids=job_input_file_ids, output_dir=OUTPUT_FOLDER_PATH)
    
    print("-" * 80)
    print(f"Script completed. Check the output folder '{OUTPUT_FOLDER_PATH}' for the results.")
    exit(0)


if __name__ == "__main__":
    main()
    # from retrieval import retrieve_and_answer

    # result = retrieve_and_answer(
    #     "output/processed/Paytm_Financial_Results_Q2_FY2024-62664f01.pdf_extraction.json",
    #     "What was the standalone total expense and loss before tax at the year end?",
    #     max_sections=5,
    # )
    # print(result["answer"])
    # print(result["reasoning"])