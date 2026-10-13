"""Undeploy a single model by name."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from trending_deploy.deploy import undeploy_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete the inference endpoint for a single model.")
    parser.add_argument("model_name", help="Model ID on the Hub, e.g. 'google-bert/bert-base-uncased'")
    args = parser.parse_args()

    success = undeploy_model(args.model_name)
    raise SystemExit(0 if success else 1)
