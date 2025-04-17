
from typing import List, Literal

from trending_deploy.constants import Model


def load_deployed_models() -> List[str]:
    """
    Load the list of models that are already deployed.
    """
    # TODO
    return []


def deploy_model(model_name: str) -> bool:
    """
    Deploy the specified model.

    Args:
        model_name (str): The name of the model to deploy.

    Returns:
        bool: True if the model was successfully deployed, False otherwise.
    """
    # TODO
    return False


def undeploy_model(model_name: str):
    """
    Undeploy the specified model.

    Args:
        model_name (str): The name of the model to undeploy.

    Returns:
        bool: True if the model was successfully undeployed, False otherwise.
    """
    # TODO
    return False


def deploy_selected_models(models: List[Model]) -> dict[Literal["deployed_success", "deployed_failed", "undeployed_success", "undeployed_failed"], str]:
    """
    Deploy the selected models.

    Args:
        models (list[Model]): A list of selected models to deploy.
    """
    to_deploy_model_names = {model.model_info.id for model in models}
    deployed_model_names = set(load_deployed_models())

    deployed_success = [[], []]
    for model_to_deploy in to_deploy_model_names - deployed_model_names:
        success = deploy_model(model_to_deploy)
        deployed_success[success].append(model_to_deploy)
    
    undeployed_success = [[], []]
    for model_to_undeploy in deployed_model_names - to_deploy_model_names:
        success = undeploy_model(model_to_undeploy)
        undeployed_success[success].append(model_to_undeploy)

    return {
        "deployed_success": deployed_success[1],
        "deployed_failed": deployed_success[0],
        "undeployed_success": undeployed_success[1],
        "undeployed_failed": undeployed_success[0],
    }