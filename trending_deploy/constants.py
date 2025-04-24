
from dataclasses import asdict, dataclass
from huggingface_hub import ModelInfo

@dataclass
class Instance:
    description: str
    memory_usage_bytes: int
    hourly_rate: float

@dataclass
class Model:
    model_info: ModelInfo
    viable_instance: Instance
    reward: float | None = None
    cost: float | None = None

    def to_dict(self):
        return asdict(self)

# Based on https://github.com/huggingface/hub-docs/tree/main/docs/inference-providers/tasks
ALL_TASKS = [
    # Audio
    "audio-classification",
    "automatic-speech-recognition",

    # Image
    "image-classification",
    "image-segmentation",
    "image-to-image",
    "object-detection",
    "text-to-image",

    # Video
    "text-to-video",

    # Text
    "feature-extraction",
    "fill-mask",
    "sentence-similarity",
    "question-answering",
    "summarization",
    "text-classification",
    "text-generation",
    "text-ranking",
    "token-classification",
    "translation",
    "zero-shot-classification",

    # Table
    "table-question-answering",
]
GPU_ONLY_TASKS = [
    "image-to-image",
    "text-to-image",
    "text-to-video",
    "text-generation",
    "automatic-speech-recognition",
    "object-detection",
]
NOT_IMPLEMENTED_TASKS = [
    "text-ranking",
]
DEFAULT_TASKS = [
    task for task in ALL_TASKS if task not in (GPU_ONLY_TASKS + NOT_IMPLEMENTED_TASKS)
]

# Just AWS for now
INSTANCES = [
    Instance(description="Intel Sapphire Rapids, 1 vCPU, 2GB", memory_usage_bytes=2 * (1024 ** 3), hourly_rate=0.033),
    Instance(description="Intel Sapphire Rapids, 2 vCPU, 4GB", memory_usage_bytes=4 * (1024 ** 3), hourly_rate=0.067),
    Instance(description="Intel Sapphire Rapids, 4 vCPU, 8GB", memory_usage_bytes=8 * (1024 ** 3), hourly_rate=0.134),
    Instance(description="Intel Sapphire Rapids, 8 vCPU, 16GB", memory_usage_bytes=16 * (1024 ** 3), hourly_rate=0.268),
    Instance(description="Intel Sapphire Rapids, 16 vCPU, 32GB", memory_usage_bytes=32 * (1024 ** 3), hourly_rate=0.536),
]
MEMORY_USAGE_TO_INSTANCE = {
    instance.memory_usage_bytes: instance for instance in sorted(INSTANCES, key=lambda x: x.memory_usage_bytes)
}