from dataclasses import dataclass
from typing import Optional, Dict, Any
import copy
import yaml

GLOBAL_VERBOSE = False

def vprint(*args, **kwargs):
    if GLOBAL_VERBOSE:
        print(*args, **kwargs)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dict `base` with `override`.
    Values in `override` win. Modifies a copy of base, returns it.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result

@dataclass
class WandbConfig:
    enabled: bool
    project: str
    run_name: Optional[str]
    custom_logging_function: Optional[str]

@dataclass
class DataConfig:
    input_dir: str
    glob_pattern: str


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    strategy: str = "default"  # "default" or "custom"
    custom_chunker_module: Optional[str] = None
    custom_chunker_fn: Optional[str] = None
    params: Dict[str, Any] = None 


@dataclass
class EmbeddingConfig:
    model_name: str
    device: str


@dataclass
class RetrieverConfig:
    top_k: int


@dataclass
class VectorStoreConfig:
    uri: str
    drop_old: bool


@dataclass
class LLMConfig:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    pipeline_task: str = "text-generation"
    response_format: Optional[str] = None
    device: Optional[str] = None
    seed: Optional[int] = None

@dataclass
class PromptConfig:
    use_system_message: bool
    system_message: str
    template: str


@dataclass
class TestDataConfig:
    path: str
    loader: str  # "jsonl" or "csv"


@dataclass
class PreprocessConfig:
    module: str
    function: str
    params: Dict[str, Any] = None


@dataclass
class MetricConfig:
    module: str
    function: str
    params: Dict[str, Any] = None

@dataclass
class LocalDumpConfig:
    enabled: bool = True
    output_dir: str = "outputs"
    filename: str = "results.jsonl"

@dataclass
class RAGConfig:
    verbose: bool
    wandb: WandbConfig
    data: DataConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retriever: RetrieverConfig
    vector_store: VectorStoreConfig
    llm: LLMConfig
    prompt: PromptConfig
    test_data: TestDataConfig
    preprocessing: PreprocessConfig
    metrics: MetricConfig
    local_dump: LocalDumpConfig


def load_config(
    default_path: str = "configs/default.yaml",
    task_path: Optional[str] = None,
) -> RAGConfig:
    print(f"[CONFIG] Loading default config from {default_path}...")
    with open(default_path, "r") as f:
        raw_default = yaml.safe_load(f)

    raw_task = {}
    if task_path is not None:
        print(f"[CONFIG] Loading task config from {task_path}...")
        with open(task_path, "r") as f:
            raw_task = yaml.safe_load(f) or {}

    # task overrides default
    merged = deep_update(raw_default, raw_task)

    merged.setdefault("chunking", {})
    merged["chunking"].setdefault("params", {})

    merged.setdefault("preprocessing", {})
    merged["preprocessing"].setdefault("params", {})

    merged.setdefault("metrics", {})
    merged["metrics"].setdefault("params", {})

    merged.setdefault("local_dump", {})

    cfg = RAGConfig(
        verbose=merged.get("verbose", False),
        wandb=WandbConfig(**merged["wandb"]),
        data=DataConfig(**merged["data"]),
        chunking=ChunkingConfig(**merged["chunking"]),
        embedding=EmbeddingConfig(**merged["embedding"]),
        retriever=RetrieverConfig(**merged["retriever"]),
        vector_store=VectorStoreConfig(**merged["vector_store"]),
        llm=LLMConfig(**merged["llm"]),
        prompt=PromptConfig(**merged["prompt"]),
        test_data=TestDataConfig(**merged["test_data"]),
        preprocessing=PreprocessConfig(**merged["preprocessing"]),
        metrics=MetricConfig(**merged["metrics"]),
        local_dump=LocalDumpConfig(**merged["local_dump"])
    )

    global GLOBAL_VERBOSE
    GLOBAL_VERBOSE = cfg.verbose

    print(
        f"[CONFIG] Loaded experiment '{cfg.wandb.run_name}' "
        f"in project '{cfg.wandb.project}' (verbose={GLOBAL_VERBOSE})."
    )
    return cfg