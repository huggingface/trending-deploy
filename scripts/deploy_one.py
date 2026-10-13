"""Deploy a single model by name."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
from huggingface_hub import model_info

from trending_deploy.constants import Instance, Model
from trending_deploy.deploy import deploy_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a single model to an inference endpoint.")
    parser.add_argument("model_name", help="Model ID on the Hub, e.g. 'google-bert/bert-base-uncased'")
    args = parser.parse_args()

    info = model_info(args.model_name, token=os.environ.get("HF_TOKEN"))
    # viable_instance is only used for memory-based instance sizing, which is
    # overridden to DEFAULT_INSTANCE_SIZE in deploy_model, so a placeholder suffices.
    dummy_instance = Instance(description="default", memory_usage_bytes=0, hourly_rate=0.0)
    model = Model(model_info=info, viable_instance=dummy_instance)

    success = deploy_model(model)
    raise SystemExit(0 if success else 1)
