
import logging
from pathlib import Path
import json
import pprint

from trending_deploy.deploy import deploy_selected_models
from trending_deploy.optimization import solve_knapsack
from trending_deploy.constants import Model, DEFAULT_TASKS
from trending_deploy.costs import get_cost_from_model
from trending_deploy.models import trending_models
from trending_deploy.rewards import get_reward_from_model


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
        # Step 1: Load the trending models as model candidates
        model_candidates: list[Model] = trending_models(tasks=self.tasks, max_models_per_task=self.max_models_per_task)

        # Step 2: Calculate the reward and cost for each model
        for model in model_candidates:
            model.reward = get_reward_from_model(model.model_info, task=model.model_info.pipeline_tag)
            model.cost = get_cost_from_model(model.model_info, task=model.model_info.pipeline_tag, instance=model.viable_instance)

        # Step 3: Solve the knapsack optimization problem given the model rewards, costs, and budget
        budget = budget or self.budget
        if budget is None:
            raise ValueError("Budget must be specified")
        
        max_reward, spent_budget, selected_models = solve_knapsack(model_candidates, budget)
        models_per_task = {task: 0 for task in self.tasks}
        for model in selected_models:
            models_per_task[model.model_info.pipeline_tag] += 1

        logging.info(f"Selected models: {len(selected_models)} out of {len(model_candidates)} candidate models")
        logging.info(f"Expected spent budget: ${spent_budget:,} out of ${budget:,}")
        logging.info(f"Maximum reward reached: {max_reward}")
        logging.info(f"Models per task:\n{pprint.pformat(models_per_task)}")

        if filename is not None:
            with open(filename, "w") as f:
                json.dump([model.to_dict() for model in selected_models], f, indent=4, default=str)

        # Step 4: Deploy the selected models
        results = deploy_selected_models(selected_models)

        logging.debug(f"Deployed models: {results['deployed_success']}")
        logging.warning(f"Failed to deploy models: {results['deployed_failed']}")
        logging.debug(f"Models already deployed: {results['undeployed_success']}")
        logging.warning(f"Failed to undeploy models: {results['undeployed_failed']}")

        return selected_models, max_reward, spent_budget

if __name__ == "__main__":
    trending = Trending(tasks=DEFAULT_TASKS, max_models_per_task=300, budget=10_000)
    selected_models, max_reward, spent_budget = trending(filename="selected_models.json")
