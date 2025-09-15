from huggingface_hub import create_scheduled_uv_job, run_uv_job
import os

BUDGET = 500
MAX_MODELS_PER_TASK = 30
HF_TOKEN = os.environ["HF_TOKEN"]
SCHEDULE = "@weekly"  # or "@daily", "@monthly", or None for one-time job

def main():
    kwargs = {
        "script_args": [
            "--budget",
            str(BUDGET),
            "--max-models-per-task",
            str(MAX_MODELS_PER_TASK),
            "--verbose",
            "--dry",
        ],
        "secrets": {"HF_TOKEN": HF_TOKEN},
        "flavor": "cpu-basic",
        "timeout": "24h",
        "namespace": "hf-inference",
    }

    if SCHEDULE is None:
        run_uv_job(
            "https://raw.githubusercontent.com/huggingface/trending-deploy/refs/heads/main/cli.py",
            **kwargs,
        )
    else:
        create_scheduled_uv_job(
            "https://raw.githubusercontent.com/huggingface/trending-deploy/refs/heads/main/cli.py",
            schedule=SCHEDULE,
            **kwargs,
        )

if __name__ == "__main__":
    main()