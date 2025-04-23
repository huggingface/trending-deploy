
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

```python
python cli.py --max-models-per-task 300 --budget 10000 --filename "selected_models" --dry
```

```
2025-03-28 16:04:44 - Selected models: 381 out of 1720 models
2025-03-28 16:04:44 - Expected spent budget: 9984 out of 10000                                                                                               
2025-03-28 16:04:44 - Maximum reward reached: 695
```