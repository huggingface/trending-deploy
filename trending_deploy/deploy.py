from typing import List, Literal, Dict, Any
import os
from huggingface_hub import (
    InferenceClient,
    create_inference_endpoint, 
    get_inference_endpoint,
    list_inference_endpoints,
    model_info as get_model_info
)

from trending_deploy.constants import Model, MEMORY_USAGE_TO_INSTANCE, Instance, INSTANCES
from trending_deploy.models import get_num_parameters_from_model, get_viable_instance_from_num_parameters

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
VENDOR = "aws"
REGION = "us-east-1"
TYPE = "public"
NAMESPACE = "hf-inference" 
ENDPOINT_PREFIX = "auto-"

# Instance size mapping based on instance memory
# Maps instance memory to HF instance size (x1, x2, etc.)
INSTANCE_SIZE_MAPPING = {
    instance.memory_usage_bytes: f"x{2**(i)}" 
    for i, instance in enumerate(sorted(INSTANCES, key=lambda x: x.memory_usage_bytes))
}


def load_deployed_models() -> List[str]:
    """
    Load the list of models that are already deployed.
    """
    try:
        endpoints = list_inference_endpoints(namespace=NAMESPACE)
        # Extract model names from endpoints starting with our prefix
        deployed_models = []
        for endpoint in endpoints:
            if endpoint.name.startswith(ENDPOINT_PREFIX):
                # Extract the model name from the repository field
                deployed_models.append(endpoint.repository)
        return deployed_models
    except Exception as e:
        print(f"Error loading deployed models: {e}")
        return []


def deploy_model(model: Model) -> bool:
    """
    Deploy the specified model.

    Args:
        model (Model): The Model object containing model_info and viable_instance.

    Returns:
        bool: True if the model was successfully deployed, False otherwise.
    """
    try:
        model_name = model.model_info.id
        endpoint_name = f"{ENDPOINT_PREFIX}{model_name.replace('/', '-')}"[:30].lower()
        
        # Get task from model info
        task = model.model_info.pipeline_tag
        
        # Get instance size directly from the viable_instance memory
        instance_size = INSTANCE_SIZE_MAPPING.get(
            model.viable_instance.memory_usage_bytes, 
            "x1"  # Default to x1 if mapping not found
        )

        if "text-embeddings-inference" in model.model_info.tags:
            endpoint = create_inference_endpoint(
                name=endpoint_name,
                namespace=NAMESPACE,
                repository=model_name,
                framework="pytorch",
                task=task,
                accelerator="cpu",
                vendor=VENDOR,
                region=REGION,
                type=TYPE,
                instance_size=instance_size,
                instance_type="intel-icl",
                scale_to_zero_timeout=15,
                domain="api-inference.endpoints.huggingface.tech",
                path=f"/models/{model_name}",
                tags=["auto", "api-inference"]
                custom_image={
                    "health_route": "/health",  # Default health route for TEI
                    "port": 80,               # Default port
                    "url": "ghcr.io/huggingface/text-embeddings-inference:latest", # Standard TEI image URL
                    "maxBatchTokens": 16384,
                    "maxConcurrentRequests": 512,
                }
                )
                # Wait for deployment (with timeout to avoid blocking indefinitely)
            endpoint.wait(timeout=300)
            return True
        else:
            endpoint = create_inference_endpoint(
                name=endpoint_name,
                namespace=NAMESPACE,
                repository=model_name,
                framework="pytorch",
                task=task,
                accelerator="cpu",
                vendor=VENDOR,
                region=REGION,
                type=TYPE,
                instance_size=instance_size,
                instance_type="intel-icl",
                scale_to_zero_timeout=15,
                domain="api-inference.endpoints.huggingface.tech",
                path=f"/models/{model_name}",
                tags=["auto", "api-inference"]
            )
            # Wait for deployment (with timeout to avoid blocking indefinitely)
            endpoint.wait(timeout=300)
            return True
    except Exception as e:
        print(f"Error deploying model {model.model_info.id}: {e}")
        return False


def undeploy_model(model_name: str) -> bool:
    """
    Undeploy the specified model.

    Args:
        model_name (str): The name of the model to undeploy.

    Returns:
        bool: True if the model was successfully undeployed, False otherwise.
    """
    try:
        # Find the endpoint for this model
        endpoints = list_inference_endpoints(namespace=NAMESPACE)
        for endpoint in endpoints:
            if endpoint.repository == model_name and endpoint.name.startswith(ENDPOINT_PREFIX):
                endpoint.delete()
                return True
        return False
    except Exception as e:
        print(f"Error undeploying model {model_name}: {e}")
        return False


def deploy_selected_models(models: List[Model]) -> dict[Literal["deployed_success", "deployed_failed", "undeployed_success", "undeployed_failed"], List[str]]:
    """
    Deploy the selected models.

    Args:
        models (list[Model]): A list of selected models to deploy.
    
    Returns:
        dict: A dictionary containing lists of successfully and unsuccessfully deployed and undeployed models.
    """
    to_deploy_models = {model.model_info.id: model for model in models}
    deployed_model_names = set(load_deployed_models())

    deployed_success = []
    deployed_failed = []
    for model_id in set(to_deploy_models.keys()) - deployed_model_names:
        success = deploy_model(to_deploy_models[model_id])
        if success:
            deployed_success.append(model_id)
        else:
            deployed_failed.append(model_id)
    
    undeployed_success = []
    undeployed_failed = []
    for model_to_undeploy in deployed_model_names - set(to_deploy_models.keys()):
        success = undeploy_model(model_to_undeploy)
        if success:
            undeployed_success.append(model_to_undeploy)
        else:
            undeployed_failed.append(model_to_undeploy)

    return {
        "deployed_success": deployed_success,
        "deployed_failed": deployed_failed,
        "undeployed_success": undeployed_success,
        "undeployed_failed": undeployed_failed,
    }