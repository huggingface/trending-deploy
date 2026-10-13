from collections import defaultdict
from collections.abc import Collection
from datetime import datetime, timezone

from huggingface_hub import HfApi, ModelInfo


COLLECTION_SLUG = "hf-inference/deployed-models-680a42b770e6b6cd546c3fbc"
NAMESPACE = "hf-inference"
ENDPOINT_PREFIX = "auto-"
TASK_TITLE_PREFIX = "Deployed Models: "


def release_date(model: ModelInfo) -> datetime:
    date = model.created_at
    if date is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.astimezone(timezone.utc)


def release_note(model: ModelInfo) -> str:
    if model.created_at is None:
        return "Release date: unknown"
    return f"Release date: {release_date(model):%Y-%m-%d}"


def deployed_models_by_task(api: HfApi, *, excluded_model_ids: Collection[str] = ()) -> dict[str, list[ModelInfo]]:
    model_ids = {
        endpoint.repository
        for endpoint in api.list_inference_endpoints(namespace=NAMESPACE)
        if endpoint.name.startswith(ENDPOINT_PREFIX)
        and endpoint.status in {"running", "scaledToZero", "paused", "updating"}
    }
    model_ids.difference_update(excluded_model_ids)
    metadata = {
        model.id: model
        for model in api.list_models(inference_provider="hf-inference", expand=["createdAt", "pipeline_tag"])
        if model.id in model_ids
    } if model_ids else {}
    grouped = defaultdict(list)
    for model_id in sorted(model_ids):
        model = metadata.get(model_id)
        if model is None:
            model = api.model_info(model_id, expand=["createdAt", "pipeline_tag"])
        if not model.pipeline_tag:
            raise ValueError(f"No task found for deployed model {model_id}")
        grouped[model.pipeline_tag].append(model)
    for models in grouped.values():
        models.sort(key=lambda model: model.id)
        models.sort(key=release_date, reverse=True)
    return dict(sorted(grouped.items()))


def sync_items(api: HfApi, slug: str, desired: list[tuple[str, str, str]]) -> None:
    collection = api.get_collection(slug)
    keys = {(item_type, item_id) for item_type, item_id, _ in desired}
    existing = {(item.item_type, item.item_id) for item in collection.items}
    for item_type, item_id, note in desired:
        if (item_type, item_id) not in existing:
            api.add_collection_item(slug, item_id=item_id, item_type=item_type, note=note, exists_ok=True)
    for item in collection.items:
        if (item.item_type, item.item_id) not in keys:
            api.delete_collection_item(slug, item_object_id=item.item_object_id)

    items = sorted(api.get_collection(slug).items, key=lambda item: item.position)
    by_key = {(item.item_type, item.item_id): item for item in items}
    for position, (item_type, item_id, note) in enumerate(desired):
        item = by_key[item_type, item_id]
        current_position = items.index(item)
        if current_position != position or item.note != note:
            api.update_collection_item(slug, item_object_id=item.item_object_id, position=position, note=note)
            items.pop(current_position)
            items.insert(position, item)


def sync_deployed_collections(*, dry_run: bool = False, excluded_model_ids: Collection[str] = ()) -> None:
    api = HfApi()
    grouped = deployed_models_by_task(api, excluded_model_ids=excluded_model_ids)
    if dry_run:
        for task, models in grouped.items():
            print(f"{TASK_TITLE_PREFIX}{task.replace('-', ' ').title()} ({len(models)} models)")
            for model in models:
                print(f"  {model.id}: {release_note(model)}")
        return

    parent = api.get_collection(COLLECTION_SLUG)
    collections = {collection.title: collection for collection in api.list_collections(owner=NAMESPACE)}
    children = []
    active_slugs = set()
    for task, models in grouped.items():
        title = TASK_TITLE_PREFIX + task.replace("-", " ").title()
        description = "Deployed on HF Inference. Newest releases first. Release dates use Hub repository creation dates."
        collection = collections.get(title)
        if collection is None:
            collection = api.create_collection(title, namespace=NAMESPACE, description=description, exists_ok=True)
        elif collection.description != description:
            api.update_collection_metadata(collection.slug, description=description)
        sync_items(api, collection.slug, [("model", model.id, release_note(model)) for model in models])
        active_slugs.add(collection.slug)
        children.append(("collection", collection.slug, f"{len(models)} deployed models. Newest releases first."))

    for collection in collections.values():
        if collection.title.startswith(TASK_TITLE_PREFIX) and collection.slug not in active_slugs:
            sync_items(api, collection.slug, [])

    # Replace the flat list only after all task collections have synced successfully.
    sync_items(api, parent.slug, children)
    description = "Models currently deployed on HF Inference, organized by task. Newest releases first within each task."
    if parent.description != description:
        api.update_collection_metadata(parent.slug, description=description)
