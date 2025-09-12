# /// script
# dependencies = [
#     "trending_deploy @ git+https://github.com/tomaarsen/trending-deploy.git@scheduled_hf_jobs",
# ]
# ///

import argparse
import logging

from trending_deploy.main import Trending


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy trending models based on optimization criteria"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="List of tasks to consider. If not provided, defaults to predefined tasks."
    )
    parser.add_argument(
        "--max-models-per-task",
        type=int,
        default=300,
        help="Maximum number of models to consider per task. Default: 200"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10_000,
        help="Budget for model deployment in monthly dollar spend. Default: 1000"
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Path to save selected models as JSON. If not provided, models won't be saved to file."
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Run in dry run mode. No models will be deployed."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level
    )

    # Initialize Trending with provided arguments
    trending = Trending(
        tasks=args.tasks,
        max_models_per_task=args.max_models_per_task,
        budget=args.budget
    )

    # Run the trending model selection and deployment
    selected_models, max_reward, spent_budget = trending(filename=args.filename, deploy_models=not args.dry)

    logging.info(f"Selected {len(selected_models)} models with total reward of {max_reward}")
    logging.info(f"Spent budget: ${spent_budget:,} out of ${args.budget:,}")

if __name__ == "__main__":
    main()
