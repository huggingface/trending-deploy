from typing import List, Literal, Dict, Any
import os
from huggingface_hub import (
    InferenceClient,
    create_inference_endpoint,
    get_inference_endpoint,
    get_collection,
    list_inference_endpoints,
    model_info as get_model_info,
    add_collection_item,
    delete_collection_item,
)

from trending_deploy.constants import Model, MEMORY_USAGE_TO_INSTANCE, Instance, INSTANCES
from trending_deploy.models import get_num_parameters_from_model, get_viable_instance_from_num_parameters

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
VENDOR = "aws"
REGION = "us-east-1"
TYPE = "protected"
NAMESPACE = "hf-inference"
ENDPOINT_PREFIX = "auto-"
COLLECTION_SLUG = "hf-inference/deployed-models-680a42b770e6b6cd546c3fbc"

# Instance size mapping based on instance memory
# Maps instance memory to HF instance size (x1, x2, etc.)
# Ensure INSTANCES are sorted by memory for reliable indexing later
SORTED_INSTANCES = sorted(INSTANCES, key=lambda x: x.memory_usage_bytes)
INSTANCE_SIZE_MAPPING = {
    instance.memory_usage_bytes: f"x{2**(i)}"
    for i, instance in enumerate(SORTED_INSTANCES)
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
        endpoint_name = f"{ENDPOINT_PREFIX}{model_name.split('/')[-1].replace('.', '-').replace('_', '-')}"[:31].lower()

        # Get task from model info
        task = model.model_info.pipeline_tag

        # Determine instance size
        initial_memory = model.viable_instance.memory_usage_bytes
        instance_size = INSTANCE_SIZE_MAPPING.get(initial_memory, "x1") # Default to x1

        # Increase instance size by one notch for text-embeddings-inference
        # With custom images for embedding models, we might not need this anymore
        if "text-embeddings-inference" in model.model_info.tags:
            instance_size = increase_instance_size(model, instance_size, initial_memory)

        endpoint_kwargs = {
            "name": endpoint_name,
            "namespace": NAMESPACE,
            "repository": model_name,
            "framework": "pytorch",
            "task": task,
            "accelerator": "cpu",
            "vendor": VENDOR,
            "region": REGION,
            "type": TYPE,
            "instance_size": instance_size, # Use the potentially upgraded size
            "instance_type": "intel-spr",
            "min_replica": 1,
            "scale_to_zero_timeout": None,
            "domain": "api-inference.endpoints.huggingface.tech",
            "path": f"/models/{model_name}",
            "tags": ["auto", "api-inference"]
        }

        # Add custom image config specifically for embedding models AFTER potentially upgrading instance size
        # Set custom image and secrets for specific model types
        image_version = None
        if "text-embeddings-inference" in model.model_info.tags:
            # Update task for sentence transformers
            if task == "feature-extraction" and (
                any(x in model.model_info.tags for x in ["sentence-transformers", "sentence transformers"])
                or model.model_info.library_name == "sentence-transformers"
            ):
                task = "sentence-embeddings"
            image_version = "6.2.0"
        elif task in ["token-classification", "text-classification"]:
            image_version = "6.2.2"

        # If a custom image is used, add the relevant image configuration
        if image_version is not None:
            endpoint_kwargs["custom_image"] = {
                "health_route": "/health",
                "port": 5000,
                "url": f"registry.internal.huggingface.tech/hf-endpoints/inference-pytorch-cpu:api-inference-{image_version}",
            }
            endpoint_kwargs["env"] = {
                "API_INFERENCE_COMPAT": "true",
                "HF_MODEL_DIR": "/repository",
                "HF_TASK": task,
            }
            endpoint_kwargs["task"] = task

        print(f"Creating endpoint {endpoint_name} for model {model_name} with instance size {instance_size}...")
        endpoint = create_inference_endpoint(**endpoint_kwargs)

        print(f"Waiting for endpoint {endpoint_name} to be ready...")
        # Wait for deployment (with timeout to avoid blocking indefinitely)
        endpoint.wait(timeout=300)
        print(f"Endpoint {endpoint_name} for model {model_name} deployed successfully.")
        add_collection_item(COLLECTION_SLUG, item_id=model_name, item_type="model")

        return True
    except Exception as e:
        print(f"Error deploying model {model.model_info.id}: {e}")
        return False

def increase_instance_size(model: Model, instance_size, initial_memory) -> bool:
    model_name = model.model_info.id
    current_index = -1
    for i, instance in enumerate(SORTED_INSTANCES):
        if instance.memory_usage_bytes == initial_memory:
            current_index = i
            break

    if current_index != -1 and current_index + 1 < len(SORTED_INSTANCES):
        next_memory = SORTED_INSTANCES[current_index + 1].memory_usage_bytes
        upgraded_size = INSTANCE_SIZE_MAPPING.get(next_memory)
        if upgraded_size:
            print(f"Upgrading instance size for TEI model {model_name} from {instance_size} to {upgraded_size}")
            instance_size = upgraded_size
        else:
            print(f"Warning: Could not find mapping for next instance size ({next_memory} bytes) for TEI model {model_name}. Using {instance_size}.")
    elif current_index != -1:
        print(f"Warning: TEI model {model_name} is already on the largest instance size ({instance_size}). Cannot upgrade further.")
    return instance_size


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
        endpoint_found = False
        for endpoint in endpoints:
            if endpoint.repository == model_name and endpoint.name.startswith(ENDPOINT_PREFIX):
                print(f"Deleting endpoint {endpoint.name} for model {model_name}...")
                endpoint.delete()
                endpoint_found = True
                print(f"Endpoint {endpoint.name} deleted successfully.")
                break # Assuming only one endpoint per model

        if not endpoint_found:
            print(f"Warning: Endpoint for model {model_name} not found. Cannot undeploy.")
            # Decide if this should be considered a failure or success if no endpoint existed
            return True # Or False, depending on desired behavior

        # Find and delete the corresponding item in the collection
        try:
            collection = get_collection(COLLECTION_SLUG)
            item_object_id_to_delete = None
            for item in collection.items:
                # item_id is the model repo id (e.g., 'bert-base-uncased')
                # _id is the collection item's internal object id
                if item.item_type == "model" and item.item_id == model_name:
                    item_object_id_to_delete = item.item_object_id
                    break

            if item_object_id_to_delete:
                print(f"Deleting item for {model_name} (ID: {item_object_id_to_delete}) from collection {COLLECTION_SLUG}...")
                delete_collection_item(COLLECTION_SLUG, item_object_id=item_object_id_to_delete)
                print(f"Collection item for {model_name} deleted successfully.")
            else:
                print(f"Warning: Could not find item for model {model_name} in collection {COLLECTION_SLUG}.")

        except Exception as e_coll:
            print(f"Error managing collection item for model {model_name}: {e_coll}")

        return True # Return True as endpoint deletion was the primary goal
    except Exception as e:
        print(f"Error during undeployment process for model {model_name}: {e}")
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