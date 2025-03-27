import glob
import json
import os
import time
import numpy as np
from tqdm import trange

def solve_knapsack(costs, rewards, budget):
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

    for model_idx in trange(1, num_models + 1):
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

'''
# This is some example usage of the function above
# costs = [2, 3, 4, 5]
# rewards = [10, 20, 30, 40]
# budget = 10
costs = [4, 3, 2, 5] * 1000
rewards = np.random.randint(10, 100, size=len(costs))
budget = 583

start_time = time.time()
max_reward, selected_items = solve_knapsack(costs, rewards, budget)
end_time = time.time()
print("Maximum Reward:", max_reward)
print("Total Cost:", sum(costs[i] for i in selected_items))
print("Selected Items:", np.array(selected_items), "with a size of", len(selected_items), "out of a total of", len(costs), "models")
print(f"Time taken: {end_time - start_time:.4f}s")
'''

trending_data = []
for trending_data_file in glob.glob("trending_data/*.json"):
    print(f"Processing {trending_data_file}...")
    with open(trending_data_file, "r") as f:
        trending_data += json.load(f)

costs = [round(model["data"]["hourly_rate"] * 30 * 24) for model in trending_data]
rewards = [model["trending_score"] for model in trending_data]
budget = 1000

start_time = time.time()
max_reward, selected_items = solve_knapsack(costs, rewards, budget)
end_time = time.time()
print("Maximum Reward:", max_reward)
print("Total Cost:", sum(costs[i] for i in selected_items))
print("Selected Items:", np.array(selected_items), "with a size of", len(selected_items), "out of a total of", len(costs), "models")
print(f"Time taken: {end_time - start_time:.4f}s")

for pipeline_tag in set(model["pipeline_tag"] for model in trending_data):
    print(f"Fetching trending models for pipeline tag: {pipeline_tag}")
    models = [model for idx, model in enumerate(trending_data) if model["pipeline_tag"] == pipeline_tag and idx in selected_items]
    
    os.makedirs("trending_data_selected", exist_ok=True)
    
    with open(f"trending_data_selected/{pipeline_tag}.json", "w") as f:
        json.dump(models, f, indent=2)

