
from typing import Iterator, List
from huggingface_hub import list_models, ModelInfo, hf_hub_download, model_info as get_model_info
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
import json
from tqdm import tqdm, trange

from trending_deploy.constants import Instance, Model, MEMORY_USAGE_TO_INSTANCE


def trending_models(tasks: list[str], max_models_per_task: int = 200) -> List[Model]:
    """
    Fetches the trending models for the specified tasks.

    Args:
        tasks (list[str] | None): A list of task names. If None, defaults to DEFAULT_TASKS.
        max_models_per_task (int): The maximum number of models to fetch per task.
        budget (int | None): The budget for the tasks in monthly dollar spend.

    Returns:
        List[Model]: A list of Model objects containing model information and viable instance.
    """
    models_to_consider: list[Model] = []
    tasks_iterator = tqdm(tasks, leave=False)
    for task in tasks_iterator:
        tasks_iterator.set_description(f"Loading trending models for {task}")
        models_to_consider.extend(trending_models_for_task(task, max_models_per_task))

    return models_to_consider

def trending_models_for_task(task: str, max_models_per_task: int = 200) -> List[Model]:
    """
    Fetches the trending models for a specific task.

    Args:
        task (str): The task for which to fetch trending models.
        max_models_per_task (int): The maximum number of models to fetch per task.

    Returns:
        List[Model]: A list of Model objects containing model information and viable instance.
    """
    models_to_consider: list[Model] = []
    trending_model_gen = trending_model_generator(task)
    try:
        for _ in trange(max_models_per_task, desc="Loading models", leave=False):
            models_to_consider.append(next(trending_model_gen))
    except StopIteration:
        pass
    return models_to_consider

def trending_model_generator(task: str) -> Iterator[Model]:
    def load_models_generator(task):
        """
        Interleaves a trending model generator with a downloads-based model generator,
        yielding models from both generators until both are exhausted. Once the trending
        generator fetches a model with a trending score of 0, it stops yielding from that
        generator.
        """
        list_models_kwargs = {
            "pipeline_tag": task,
            "filter": "endpoints_compatible",
            "expand": [
                "createdAt",
                "trendingScore",
                "tags",
                "library_name",
                "likes",
                "downloads",
                "downloadsAllTime",
                "safetensors",
                "pipeline_tag",
            ],
        }
        trending_models_generator = list_models(**list_models_kwargs)
        downloads_models_generator = list_models(**list_models_kwargs, sort="downloads")

        yielded_model_ids = set()
        yield_trending = True

        while True:
            if yield_trending:
                try:
                    model = next(trending_models_generator)
                except StopIteration:
                    break
                if model.trending_score == 0:
                    yield_trending = False
                else:
                    if model.id not in yielded_model_ids:
                        yielded_model_ids.add(model.id)
                        yield model

            try:
                model = next(downloads_models_generator)
            except StopIteration:
                break
            if model.id not in yielded_model_ids:
                yielded_model_ids.add(model.id)
                yield model

    for model_info in load_models_generator(task):
        if "custom_code" in model_info.tags:
            continue

        # Get the number of parameters to determine which instance type is viable.
        # Sometimes it may fail (e.g. non-authorized models), so we just skip those models.
        try:
            num_parameters = get_num_parameters_from_model(model_info)
        except Exception:
            continue
        if num_parameters is None:
            continue
        
        viable_instance: Instance = get_viable_instance_from_num_parameters(num_parameters)
        if viable_instance is None:
            continue

        yield Model(model_info=model_info, viable_instance=viable_instance)

def get_num_parameters_from_model(model: ModelInfo):
    safetensors = model.safetensors
    if safetensors:
        return safetensors.total

    bytes_per_param = 4
    files = get_model_info(model.id, files_metadata=True).siblings
    for file in files:
        if file.rfilename == "pytorch_model.bin":
            return file.size // bytes_per_param

        if file.rfilename == "pytorch_model.bin.index.json":
            disable_progress_bars()
            index_path = hf_hub_download(model.id, filename="pytorch_model.bin.index.json")
            enable_progress_bars()
            """
            {
            "metadata": {
                "total_size": 28272820224
            },....
            """
            index = json.load(open(index_path))
            if ("metadata" in index) and ("total_size" in index["metadata"]):
                return index["metadata"]["total_size"] // bytes_per_param

    return None

def get_viable_instance_from_num_parameters(num_parameters: int):
    model_memory_usage_bytes = num_parameters * 4
    memory_factor = 2.2
    viable_instance = None
    for max_instance_memory_usage, instance in MEMORY_USAGE_TO_INSTANCE.items():
        if model_memory_usage_bytes * memory_factor < max_instance_memory_usage:
            viable_instance = instance
            break
    return viable_instance

