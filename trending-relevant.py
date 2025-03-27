import requests
import json
from datetime import datetime
import os
import time

list_of_tasks = ['feature-extraction', 
                 'fill-mask', 
                 'token-classification',]

# Create a directory to store the JSON files if it doesn't exist
output_dir = "trending_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get current date for the filename
current_date = datetime.now().strftime("%Y-%m-%d")

# Get Hugging Face token from environment variable or use a default value
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Function to fetch trending models for a specific task
def fetch_trending_models(task, limit=10):
    url = f"https://huggingface.co/api/models"
    params = {
        "limit": limit,
        "pipeline_tag": task,
        "sort": "likes30d",
        "full": "True",
        "config": "True"
    }
    
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for task '{task}': {e}")
        return None

# Iterate through all tasks and save results to JSON files
for task in list_of_tasks:
    print(f"Fetching trending models for task: {task}")
    
    # Fetch trending models for the current task
    trending_models = fetch_trending_models(task)
    
    if trending_models is not None:
        # Create filename with task and date
        filename = f"{output_dir}/{task}_{current_date}.json"
        
        # Save the data to a JSON file
        with open(filename, 'w') as f:
            json.dump(trending_models, f, indent=2)
        
        print(f"Saved trending models for '{task}' to {filename}")
    
    # Add a small delay to avoid hitting rate limits
    time.sleep(0.5)

print("All tasks completed!")