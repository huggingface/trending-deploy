"""
The Python script variant of the big red shut-down button for the trending-deploy package.
This script undeploys all models that have previously been deployed.
"""

from trending_deploy.deploy import load_deployed_models, undeploy_model

if __name__ == "__main__":
    # Load the deployed models
    deployed_models = load_deployed_models()
    
    # Undeploy all deployed models
    for model in deployed_models:
        success = undeploy_model(model)
        if success:
            print(f"Successfully undeployed model: {model}")
        else:
            print(f"Failed to undeploy model: {model}")
