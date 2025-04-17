

from typing import List

import numpy as np
from tqdm import trange

from trending_deploy.constants import Model


def solve_knapsack(model_candidates: List[Model], budget: int) -> tuple[float, int, List[Model]]:
    """
    Solve the 0/1 Knapsack Problem using dynamic programming with NumPy.

    Args:
        model_candidates (List[Model]): A list of model candidates.
        budget (int): The maximum total cost that we can allocate.
    
    Returns:
        tuple: A tuple containing the maximum reward, the spent budget, and the selected models.
    """
    # Extract costs and rewards from the model candidates
    costs: List[int] = [model.cost for model in model_candidates]
    rewards: List[float] = [model.reward for model in model_candidates]
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
    selected_models = []
    budget_idx = budget
    for model_idx in range(num_models, 0, -1):
        if rewards_table[model_idx, budget_idx] != rewards_table[model_idx - 1, budget_idx]:
            selected_models.append(model_candidates[model_idx - 1])
            budget_idx -= costs[model_idx - 1]
    spent_budget = budget - budget_idx

    return max_reward, spent_budget, selected_models[::-1]