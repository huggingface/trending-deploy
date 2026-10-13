"""Preview or apply the task collection layout for existing deployments."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trending_deploy.collections import sync_deployed_collections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Update the collections on the Hub.")
    args = parser.parse_args()
    sync_deployed_collections(dry_run=not args.apply)
