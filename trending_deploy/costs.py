


from huggingface_hub import ModelInfo
from trending_deploy.constants import Instance


def get_cost_from_model(model: ModelInfo, task: str, instance: Instance):
    # We can use likes, downloads, task-specific multipliers, but also just the hourly rate
    # of the instance multiplied by the ~number of hours in a month (720)
    return round(instance.hourly_rate * 24 * 30)