from huggingface_hub import create_scheduled_uv_job
import os

BUDGET = 500
MAX_MODELS_PER_TASK = 30
HF_TOKEN = os.environ["HF_TOKEN"]

def main():
    create_scheduled_uv_job(
        "https://raw.githubusercontent.com/tomaarsen/trending-deploy/refs/heads/scheduled_hf_jobs/cli.py",
        script_args=[
            "--budget",
            str(BUDGET),
            "--max-models-per-task",
            str(MAX_MODELS_PER_TASK),
            "--verbose",
            "--dry",
        ],
        schedule="@weekly",
        secrets={"HF_TOKEN": HF_TOKEN},
        flavor="cpu-basic",
        timeout="8h",
        namespace="hf-inference",
    )

if __name__ == "__main__":
    main()