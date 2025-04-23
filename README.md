
## Deployment
Given a set of Machine Learning models with trending scores and hosting costs, determine the optimal subset of models to deploy within a limited budget to maximize total reward.

### Key Components:

* **Rewards:** 1) Trending scores of ML models, 2) Task multipliers
   * Gather trending scores from API, e.g. by loading the top _n_ trending models per task (note: best to vary per task). Also determine the model size.
* **Costs:** Integers representing hosting costs of CPU models
   * Create a function mapping model size to hardware (cost). We have to turn these into integers (e.g. rounded monthly $USD) for most Knapsack solver implementations to work nicely.
* **Constraint:** Limited budget (knapsack capacity)

**Goal:** Solve the [0/1 Knapsack optimization problem](https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem) to maximize total reward while staying within the budget.

### Sample Usage

```bash
python cli.py --max-models-per-task 300 --budget 10000 --filename "selected_models" --dry
```

```
2025-04-23 15:27:18 - Selected models: 391 out of 2021 candidate models
2025-04-23 15:27:18 - Expected spent budget: $9,984 out of $10,000                                                                                                                           
2025-04-23 15:27:18 - Maximum reward reached: 2063
2025-04-23 15:27:18 - Models per task:
{'feature-extraction': 69,
 'fill-mask': 99,
 'sentence-similarity': 67,
 'text-classification': 86,
 'token-classification': 33,
 'translation': 26,
 'zero-shot-classification': 11}
2025-04-23 15:27:34 - Selected 391 models with total reward of 2063
2025-04-23 15:27:34 - Spent budget: $9,984 out of $10,000
```