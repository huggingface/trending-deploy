import math
from huggingface_hub import ModelInfo


def get_reward_from_model(model: ModelInfo, task: str):
    # We can use trending score as a reward, but we can also include likes, downloads, 
    # task-specific multipliers, etc.
    if model.downloads < 1000:
        return 0  # Filter out models with less than 1000 monthly downloads
    return model.trending_score + math.log(model.likes + 1)