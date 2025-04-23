
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
python cli.py --max-models-per-task 300 --budget 10000 --filename "selected_models.json" --dry
```

```
2025-04-23 15:54:38 - Selected models: 389 out of 4211 candidate models
2025-04-23 15:54:38 - Expected spent budget: $9,985 out of $10,000
2025-04-23 15:54:38 - Maximum reward reached: 2239
2025-04-23 15:54:38 - Models per task:
{'audio-classification': 6,
 'automatic-speech-recognition': 16,
 'feature-extraction': 54,
 'fill-mask': 99,
 'image-classification': 29,
 'image-segmentation': 17,
 'object-detection': 12,
 'question-answering': 10,
 'sentence-similarity': 65,
 'summarization': 6,
 'table-question-answering': 1,
 'text-classification': 47,
 'token-classification': 10,
 'translation': 13,
 'zero-shot-classification': 4}
2025-04-23 15:54:38 - Selected 389 models with total reward of 2239
2025-04-23 15:54:38 - Spent budget: $9,985 out of $10,000
```