from dataclasses import asdict
from huggingface_hub import list_models, hf_hub_url, get_hf_file_metadata, ModelInfo, hf_hub_download
import json
from datetime import datetime
import os
import time

list_of_tasks = [
    "feature-extraction",
    "sentence-similarity",
    "fill-mask",
    "token-classification",
]

# Create a directory to store the JSON files if it doesn't exist
output_dir = "trending_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get current date for the filename
current_date = datetime.now().strftime("%Y-%m-%d")

# Just AWS for now
memory_usage_to_instance = {
    2 * (1024 ** 3): {
        "description": "Intel Sapphire Rapids, 1 vCPU, 2GB",
        "hourly_rate": 0.033,
    },
    4 * (1024 ** 3): {
        "description": "Intel Sapphire Rapids, 2 vCPU, 4GB",
        "hourly_rate": 0.067,
    },
    8 * (1024 ** 3): {
        "description": "Intel Sapphire Rapids, 4 vCPU, 8GB",
        "hourly_rate": 0.134,
    },
    16 * (1024 ** 3): {
        "description": "Intel Sapphire Rapids, 8 vCPU, 16GB",
        "hourly_rate": 0.268,
    },
    32 * (1024 ** 3): {
        "description": "Intel Sapphire Rapids, 16 vCPU, 32GB",
        "hourly_rate": 0.536,
    },
}

def get_num_parameters(model: ModelInfo):
    safetensors = model.safetensors
    if safetensors:
        return safetensors.total

    bytes_per_param = 4
    if "pytorch_model.bin" in model.siblings:
        url = hf_hub_url(model.id, filename="pytorch_model.bin")
        meta = get_hf_file_metadata(url)
        return meta.size // bytes_per_param

    if "pytorch_model.bin.index.json" in model.siblings:
        index_path = hf_hub_download(model.id, filename="pytorch_model.bin.index.json")
        """
        {
        "metadata": {
            "total_size": 28272820224
        },....
        """
        index = json.load(open(index_path))
        if ("metadata" in index) and ("total_size" in index["metadata"]):
            return index["metadata"]["total_size"] // bytes_per_param

    return None


def get_trending_models(task):
    for model in list_models(
        pipeline_tag=task,
        tags="endpoints_compatible",
        expand=[
            "createdAt",
            "trendingScore",
            "tags",
            "library_name",
            "likes",
            "downloads",
            "downloadsAllTime",
            "safetensors",
            "siblings",
            "pipeline_tag",
        ],
    ):
        # Get the number of parameters to determine which instance type is viable
        num_parameters = get_num_parameters(model)
        if num_parameters is None:
            continue
        
        model_memory_usage_bytes = num_parameters * 4
        memory_factor = 2.2  # Memory overhead factor

        viable_instance = None
        for max_instance_memory_usage, instance in memory_usage_to_instance.items():
            if model_memory_usage_bytes * memory_factor < max_instance_memory_usage:
                viable_instance = instance
                break
        if viable_instance is None:
            continue

        model_dict = asdict(model)
        model_dict["created_at"] = model.created_at.isoformat()
        del model_dict["siblings"]
        model_dict["data"] = {
            "memory_usage": model_memory_usage_bytes,
            "instance": viable_instance["description"],
            "hourly_rate": viable_instance["hourly_rate"],
        }
        yield model_dict

limit = 200

# Iterate through all tasks and save results to JSON files
for task in list_of_tasks:
    print(f"Fetching trending models for task: {task}")

    # Fetch trending models for the current task
    trending_model_generator = get_trending_models(task)

    # Create filename with task and date
    filename = f"{output_dir}/{task}_{current_date}.json"

    # Save the data to a JSON file
    with open(filename, "w") as f:
        json.dump([next(trending_model_generator) for _ in range(limit)], f, indent=2)

    print(f"Saved trending models for '{task}' to {filename}")

    # Add a small delay to avoid hitting rate limits
    time.sleep(0.5)

print("All tasks completed!")
