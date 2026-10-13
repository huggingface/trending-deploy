import copy
from types import SimpleNamespace

import pytest
from huggingface_hub import CollectionItem, ModelInfo

from trending_deploy import collections, deploy


class FakeApi:
    def __init__(self):
        self.models = {
            "org/old": ModelInfo(id="org/old", pipeline_tag="text-classification", createdAt="2024-01-01T00:00:00Z"),
            "org/new": ModelInfo(id="org/new", pipeline_tag="text-classification", createdAt="2026-01-01T00:00:00Z"),
            "org/tie": ModelInfo(id="org/tie", pipeline_tag="text-classification", createdAt="2026-01-01T00:00:00Z"),
            "org/unknown": ModelInfo(id="org/unknown", pipeline_tag="text-classification"),
            "org/image": ModelInfo(id="org/image", pipeline_tag="image-classification", createdAt="2025-01-01T00:00:00Z"),
        }
        self.endpoints = [
            SimpleNamespace(name=f"auto-{model_id.split('/')[-1]}", repository=model_id, status="running")
            for model_id in self.models
        ]
        self.collections = {
            collections.COLLECTION_SLUG: SimpleNamespace(
                slug=collections.COLLECTION_SLUG, title="Deployed Models", description="Original description",
                upvotes=41, items=[self.item("model", "org/old", 0), self.item("model", "org/stale", 1)],
            )
        }
        self.writes = []

    def item(self, item_type, item_id, position, note=None):
        return CollectionItem(_id=f"item-{item_id}", id=item_id, type=item_type, position=position,
                              note={"text": note} if note else None)

    def list_inference_endpoints(self, namespace):
        return self.endpoints

    def model_info(self, model_id, expand):
        return self.models[model_id]

    def list_models(self, inference_provider, expand):
        return list(self.models.values())

    def get_collection(self, slug):
        return copy.deepcopy(self.collections[slug])

    def list_collections(self, owner):
        return [self.get_collection(slug) for slug in self.collections]

    def create_collection(self, title, namespace, description, exists_ok):
        slug = namespace + "/" + title.lower().replace(" ", "-")
        self.writes.append(("create", slug))
        self.collections[slug] = SimpleNamespace(slug=slug, title=title, description=description, items=[])
        return self.get_collection(slug)

    def add_collection_item(self, slug, item_id, item_type, note, exists_ok):
        self.writes.append(("add", slug))
        items = self.collections[slug].items
        items.append(self.item(item_type, item_id, len(items), note))

    def delete_collection_item(self, slug, item_object_id):
        self.writes.append(("delete", slug))
        items = self.collections[slug].items
        items[:] = [item for item in items if item.item_object_id != item_object_id]
        for position, item in enumerate(items):
            item.position = position

    def update_collection_item(self, slug, item_object_id, position, note):
        self.writes.append(("update_item", slug))
        items = self.collections[slug].items
        item = next(item for item in items if item.item_object_id == item_object_id)
        items.remove(item)
        items.insert(position, item)
        item.note = note
        for index, item in enumerate(items):
            item.position = index

    def update_collection_metadata(self, slug, description):
        self.writes.append(("metadata", slug))
        self.collections[slug].description = description


@pytest.fixture
def api(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(collections, "HfApi", lambda: api)
    return api


def test_migration_and_repeat_run(api):
    collections.sync_deployed_collections()
    parent = api.get_collection(collections.COLLECTION_SLUG)
    assert parent.upvotes == 41
    assert parent.title == "Deployed Models"
    assert [item.item_type for item in parent.items] == ["collection", "collection"]
    text = api.get_collection(parent.items[1].item_id)
    assert [item.item_id for item in text.items] == ["org/new", "org/tie", "org/old", "org/unknown"]
    assert [item.note for item in text.items] == [
        "Release date: 2026-01-01", "Release date: 2026-01-01", "Release date: 2024-01-01", "Release date: unknown",
    ]
    api.writes.clear()
    collections.sync_deployed_collections()
    assert api.writes == []


def test_reorders_existing_items_and_removes_stale_models(api):
    collections.sync_deployed_collections()
    parent = api.get_collection(collections.COLLECTION_SLUG)
    slug = parent.items[1].item_id
    api.collections[slug].items.reverse()
    for position, item in enumerate(api.collections[slug].items):
        item.position = position
        item.note = "outdated"
    api.endpoints = [endpoint for endpoint in api.endpoints if endpoint.repository != "org/old"]
    collections.sync_deployed_collections()
    text = api.get_collection(slug)
    assert [item.item_id for item in text.items] == ["org/new", "org/tie", "org/unknown"]
    assert text.items[0].note == "Release date: 2026-01-01"


def test_empty_tasks_are_unlinked_and_reused(api):
    collections.sync_deployed_collections()
    original_slugs = set(api.collections)
    endpoints = api.endpoints
    api.endpoints = []
    collections.sync_deployed_collections()
    assert all(not collection.items for collection in api.collections.values())
    api.endpoints = endpoints
    collections.sync_deployed_collections()
    assert set(api.collections) == original_slugs


def test_metadata_failure_does_not_change_collections(api, monkeypatch):
    def fail_metadata(**kwargs):
        raise RuntimeError("Metadata unavailable")

    monkeypatch.setattr(api, "list_models", fail_metadata)
    with pytest.raises(RuntimeError, match="Metadata unavailable"):
        collections.sync_deployed_collections()
    assert api.writes == []


def test_falls_back_for_models_missing_from_provider_listing(api, monkeypatch):
    monkeypatch.setattr(api, "list_models", lambda **kwargs: [])
    grouped = collections.deployed_models_by_task(api)
    assert sum(map(len, grouped.values())) == 5


def test_child_failure_preserves_flat_parent(api, monkeypatch):
    def fail_write(*args, **kwargs):
        raise RuntimeError("Write failed")

    parent = api.get_collection(collections.COLLECTION_SLUG)
    monkeypatch.setattr(api, "add_collection_item", fail_write)
    with pytest.raises(RuntimeError, match="Write failed"):
        collections.sync_deployed_collections()
    assert api.get_collection(parent.slug) == parent


def test_preview_does_not_write(api):
    collections.sync_deployed_collections(dry_run=True)
    assert api.writes == []


def test_excludes_failed_and_unmanaged_endpoints(api):
    api.endpoints.extend([
        SimpleNamespace(name="auto-failed", repository="org/not-present", status="failed"),
        SimpleNamespace(name="manual-model", repository="org/not-present", status="running"),
    ])
    grouped = collections.deployed_models_by_task(api)
    assert sum(map(len, grouped.values())) == 5


def test_batch_syncs_once_even_without_deployment_changes(monkeypatch):
    sync_calls = []
    monkeypatch.setattr(deploy, "load_deployed_models", lambda: [])
    monkeypatch.setattr(deploy, "_sync_collections", lambda **kwargs: sync_calls.append(kwargs))
    deploy.deploy_selected_models([])
    assert sync_calls == [{"excluded_model_ids": []}]


def test_undeploy_removes_model_despite_stale_endpoint_listing(api, monkeypatch):
    collections.sync_deployed_collections()
    endpoint = next(endpoint for endpoint in api.endpoints if endpoint.repository == "org/new")
    deletions = []
    endpoint.delete = lambda: deletions.append(endpoint.name)
    monkeypatch.setattr(deploy, "list_inference_endpoints", api.list_inference_endpoints)

    assert deploy.undeploy_model("org/new")

    assert deletions == [endpoint.name]
    assert endpoint in api.endpoints
    items = [item for collection in api.collections.values() for item in collection.items]
    assert not any(item.item_type == "model" and item.item_id == "org/new" for item in items)
    assert any(item.item_type == "model" and item.item_id == "org/tie" for item in items)


def test_batch_sync_excludes_successful_deletions_but_keeps_failed_deletions(api, monkeypatch):
    def fail_delete():
        raise RuntimeError("Deletion failed")

    for endpoint in api.endpoints:
        endpoint.delete = fail_delete if endpoint.repository == "org/old" else lambda: None
    monkeypatch.setattr(deploy, "list_inference_endpoints", api.list_inference_endpoints)

    result = deploy.deploy_selected_models([])

    assert result["undeployed_failed"] == ["org/old"]
    assert set(result["undeployed_success"]) == set(api.models) - {"org/old"}
    items = [item for collection in api.collections.values() for item in collection.items]
    assert [item.item_id for item in items if item.item_type == "model"] == ["org/old"]


def test_failed_undeploy_keeps_collection_entry(api, monkeypatch):
    def fail_delete():
        raise RuntimeError("Deletion failed")

    collections.sync_deployed_collections()
    original_collections = copy.deepcopy(api.collections)
    endpoint = next(endpoint for endpoint in api.endpoints if endpoint.repository == "org/new")
    endpoint.delete = fail_delete
    monkeypatch.setattr(deploy, "list_inference_endpoints", api.list_inference_endpoints)

    assert not deploy.undeploy_model("org/new")
    assert api.collections == original_collections
