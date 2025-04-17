
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

# Text only for now, we can extend to vision and beyond as well
DEFAULT_TASKS = [
    "feature-extraction",
    "sentence-similarity",
    "fill-mask",
    "token-classification",
    "text-classification",
    "zero-shot-classification",
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