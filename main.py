
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import List
from huggingface_hub import list_models, ModelInfo, hf_hub_download, model_info as get_model_info
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
import json
import json
import numpy as np
from tqdm import tqdm, trange

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


DEFAULT_TASKS = [
    "feature-extraction",
    "sentence-similarity",
    "fill-mask",
    "token-classification",
]


@dataclass
class Instance:
    description: str
    memory_usage_bytes: int
    hourly_rate: float

@dataclass
class Model:
    model_info: ModelInfo
    reward: float
    cost: float
    viable_instance: Instance

    def to_dict(self):
        return asdict(self)

# Just AWS for now
INSTANCES = [
    Instance(description="Intel Sapphire Rapids, 1 vCPU, 2GB", memory_usage_bytes=2 * (1024 ** 3), hourly_rate=0.033),
    Instance(description="Intel Sapphire Rapids, 2 vCPU, 4GB", memory_usage_bytes=4 * (1024 ** 3), hourly_rate=0.067),
    Instance(description="Intel Sapphire Rapids, 4 vCPU, 8GB", memory_usage_bytes=8 * (1024 ** 3), hourly_rate=0.134),
    Instance(description="Intel Sapphire Rapids, 8 vCPU, 16GB", memory_usage_bytes=16 * (1024 ** 3), hourly_rate=0.268),
    Instance(description="Intel Sapphire Rapids, 16 vCPU, 32GB", memory_usage_bytes=32 * (1024 ** 3), hourly_rate=0.536),
]
MEMORY_USAGE_TO_INSTANCE = {
    instance.memory_usage_bytes: instance for instance in sorted(INSTANCES, key=lambda x: x.memory_usage_bytes)
}

class Trending():
    def __init__(self, tasks: list[str] | None = None, max_models_per_task: int = 200, budget: int | None = 1000):
        """
        Initializes the instance with the specified tasks, maximum models per task, and budget.

        Args:
            tasks (list[str] | None, optional): A list of task names. If None, defaults to DEFAULT_TASKS. Defaults to None.
            max_models_per_task (int, optional): The maximum number of models allowed per task. Defaults to 200.
            budget (int | None, optional): The budget for the tasks in monthly dollar spend. Defaults to 1000.
        """
        if tasks is None:
            tasks = DEFAULT_TASKS
        self.tasks = tasks
        self.max_models_per_task = max_models_per_task
        self.budget = budget

    def __call__(self, budget: int | None = None, filename: str | Path | None = None):
        # Step 1: Load the trending models and their rewards and costs
        models_to_consider: list[Model] = []
        tasks_iterator = tqdm(self.tasks, leave=False)
        for task in tasks_iterator:
            tasks_iterator.set_description(f"Loading trending models for {task}")
            trending_model_generator = self.trending_model_generator(task)
            models_to_consider += [next(trending_model_generator) for _ in trange(self.max_models_per_task, desc="Loading models", leave=False)]

        # Step 2: Solve the knapsack problem given the rewards and costs of the models
        budget = budget or self.budget
        if budget is None:
            raise ValueError("Budget must be specified")
        
        costs = [model.cost for model in models_to_consider]
        rewards = [model.reward for model in models_to_consider]
        max_reward, selected_items = self.solve_knapsack(costs, rewards, budget)
        selected_models: list[Model] = [models_to_consider[i] for i in selected_items]
        spent_budget = sum(costs[i] for i in selected_items)

        logging.info(f"Selected models: {len(selected_models)} out of {len(models_to_consider)} models")
        logging.info(f"Expected spent budget: {spent_budget} out of {budget}")
        logging.info(f"Maximum reward reached: {max_reward}")

        if filename is not None:
            with open(filename, "w") as f:
                json.dump([model.to_dict() for model in selected_models], f, indent=4, default=str)

        return selected_models, max_reward, spent_budget

    def trending_model_generator(self, task: str):
        for model_info in list_models(
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
                "pipeline_tag",
            ],
        ):
            # Get the number of parameters to determine which instance type is viable.
            # Sometimes it may fail (e.g. non-authorized models), so we just skip those models.
            try:
                num_parameters = self.get_num_parameters_from_model(model_info)
            except Exception as exc:
                continue
            if num_parameters is None:
                continue
            
            viable_instance: Instance = self.get_viable_instance_from_num_parameters(num_parameters)
            if viable_instance is None:
                continue

            reward = self.get_reward_from_model(model_info, task)
            cost = self.get_cost_from_model(model_info, task, viable_instance)

            yield Model(model_info=model_info, reward=reward, cost=cost, viable_instance=viable_instance)

    def get_num_parameters_from_model(self, model: ModelInfo):
        safetensors = model.safetensors
        if safetensors:
            return safetensors.total

        bytes_per_param = 4
        files = get_model_info(model.id, files_metadata=True).siblings
        for file in files:
            if file.rfilename == "pytorch_model.bin":
                return file.size // bytes_per_param

            if file.rfilename == "pytorch_model.bin.index.json":
                disable_progress_bars()
                index_path = hf_hub_download(model.id, filename="pytorch_model.bin.index.json")
                enable_progress_bars()
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

    def get_viable_instance_from_num_parameters(self, num_parameters: int):
        model_memory_usage_bytes = num_parameters * 4
        memory_factor = 2.2
        viable_instance = None
        for max_instance_memory_usage, instance in MEMORY_USAGE_TO_INSTANCE.items():
            if model_memory_usage_bytes * memory_factor < max_instance_memory_usage:
                viable_instance = instance
                break
        return viable_instance
    
    def get_reward_from_model(self, model: ModelInfo, task: str):
        # We can use trending score as a reward, but we can also include likes, downloads, 
        # task-specific multipliers, etc.
        return model.trending_score
    
    def get_cost_from_model(self, model: ModelInfo, task: str, instance: Instance):
        # We can use likes, downloads, task-specific multipliers, but also just the hourly rate
        # of the instance multiplied by the ~number of hours in a month (720)
        return round(instance.hourly_rate * 24 * 30)

    def solve_knapsack(self, costs: List[int], rewards: List[float], budget: int):
        """
        Solve the 0/1 Knapsack Problem using dynamic programming with NumPy.

        Parameters:
        budget (int): The maximum total cost that we can allocate.
        costs (list): A list of costs for each item.
        rewards (list): A list of rewards for each item.

        Returns:
        max_reward (int): The maximum reward that can be achieved.
        selected_items (list): A list of indices of the selected items.
        """
        num_models = len(costs)
        assert num_models == len(rewards)

        # This can be seen as the number of models seen so far and the remaining budget
        rewards_table = np.zeros((num_models + 1, budget + 1), dtype=int)

        for model_idx in trange(1, num_models + 1, desc="Solving Model Optimization", leave=False):
            for budget_idx in range(1, budget + 1):
                if costs[model_idx - 1] <= budget_idx:
                    # If the current item's cost is within the remaining budget, choose the maximum reward between
                    # not taking the item and taking the item
                    rewards_table[model_idx, budget_idx] = max(
                        rewards_table[model_idx - 1, budget_idx],  # Not taking the item
                        rewards_table[model_idx - 1, budget_idx - costs[model_idx - 1]] + rewards[model_idx - 1]  # Taking the item
                    )
                else:
                    # If the current item's cost exceeds the remaining budget, skip the item by setting the reward to be
                    # the same as before this model was considered
                    rewards_table[model_idx, budget_idx] = rewards_table[model_idx - 1, budget_idx]

        max_reward = rewards_table[num_models, budget]
        selected_items = []
        budget_idx = budget
        for model_idx in range(num_models, 0, -1):
            if rewards_table[model_idx, budget_idx] != rewards_table[model_idx - 1, budget_idx]:
                selected_items.append(model_idx - 1)
                budget_idx -= costs[model_idx - 1]

        return max_reward, selected_items[::-1]

if __name__ == "__main__":
    trending = Trending(tasks=DEFAULT_TASKS, max_models_per_task=200, budget=1000)
    selected_models, max_reward, spent_budget = trending(filename="selected_models.json")
