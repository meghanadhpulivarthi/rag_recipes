from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List
import copy
import yaml
import re

GLOBAL_VERBOSE = False

def vprint(*args, **kwargs):
    if GLOBAL_VERBOSE:
        print(*args, **kwargs)


def substitute_templates(data: Any, defaults: Dict[str, str]) -> Any:
    """
    Recursively replace ${variable} templates in data with values from defaults.
    Works with strings, lists, dicts, and any nested combination.
    """
    if isinstance(data, str):
        # Find all ${variable} patterns and replace them
        def replace_var(match):
            var_name = match.group(1)
            return defaults.get(var_name, match.group(0))  # Keep original if not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_var, data)
    elif isinstance(data, dict):
        return {k: substitute_templates(v, defaults) for k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_templates(item, defaults) for item in data]
    else:
        return data


def add_dynamic_cli_args(parser: 'argparse.ArgumentParser', config_path: str) -> Dict[str, str]:
    """
    Add CLI arguments dynamically based on defaults section in config file.
    Returns a dictionary mapping CLI arg names to config variable names.
    """
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
        
        defaults = config_data.get("defaults", {})
        arg_mapping = {}
        
        for var_name, default_value in defaults.items():
            # Create CLI argument name (e.g., gen_cfg from defaults)
            cli_arg_name = f"--{var_name}"
            arg_mapping[cli_arg_name] = var_name
            
            # Add argument to parser
            parser.add_argument(
                cli_arg_name, 
                type=str, 
                default=default_value,
                help=f"Override default value for '{var_name}' (default: {default_value})"
            )
        
        return arg_mapping
        
    except FileNotFoundError:
        print(f"[CONFIG] Warning: Config file {config_path} not found, no dynamic CLI args added")
        return {}
    except Exception as e:
        print(f"[CONFIG] Warning: Error loading defaults from {config_path}: {e}")
        return {}


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
class VectorStoreConfig:
    uri: str
    drop_old: bool


@dataclass
class DenseRetrieverConfig:
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    type: str = "dense"
    top_k: int = 5


@dataclass
class SparseRetrieverConfig:
    dump_path: str
    type: str = "sparse"
    top_k: int = 5

@dataclass
class EnsembleRetrieverConfig:
    retrievers: List[Dict[str, Any]]
    type: str = "ensemble"
    top_k: int = 5
    weights: List[float] = None


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
    retrieval_dump_path: Optional[str] = None


@dataclass
class RAGConfig:
    verbose: bool
    wandb: WandbConfig
    data: DataConfig
    chunking: ChunkingConfig
    retriever: Union[DenseRetrieverConfig, SparseRetrieverConfig, EnsembleRetrieverConfig]
    llm: LLMConfig
    prompt: PromptConfig
    test_data: TestDataConfig
    preprocessing: PreprocessConfig
    metrics: MetricConfig
    local_dump: LocalDumpConfig

def load_config(
    default_path: str = "configs/default.yaml",
    task_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, str]] = None,
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

    # Extract defaults section (remove from merged config)
    defaults = merged.pop("defaults", {})
    
    # Apply CLI overrides to defaults (CLI has highest priority)
    if cli_overrides:
        defaults.update(cli_overrides)

    # Apply template substitution using resolved defaults
    merged = substitute_templates(merged, defaults)

    merged.setdefault("chunking", {})
    merged["chunking"].setdefault("params", {})

    merged.setdefault("preprocessing", {})
    merged["preprocessing"].setdefault("params", {})

    merged.setdefault("metrics", {})
    merged["metrics"].setdefault("params", {})

    merged.setdefault("local_dump", {})

    # Convert nested dictionaries to dataclass instances
    merged["wandb"] = WandbConfig(**merged["wandb"])
    merged["data"] = DataConfig(**merged["data"])
    merged["chunking"] = ChunkingConfig(**merged["chunking"])
    merged["llm"] = LLMConfig(**merged["llm"])
    merged["prompt"] = PromptConfig(**merged["prompt"])
    merged["test_data"] = TestDataConfig(**merged["test_data"])
    merged["preprocessing"] = PreprocessConfig(**merged["preprocessing"])
    merged["metrics"] = MetricConfig(**merged["metrics"])
    merged["local_dump"] = LocalDumpConfig(**merged["local_dump"])
    
    # Handle retriever configuration
    retriever_config = merged["retriever"]
    retriever_type = retriever_config.get("type", "dense")
    
    if retriever_type == "dense":
        # Dense retriever requires embedding and vector_store
        if "embedding" not in retriever_config:
            raise ValueError("Dense retriever requires 'embedding' configuration under retriever section")
        if "vector_store" not in retriever_config:
            raise ValueError("Dense retriever requires 'vector_store' configuration under retriever section")
        
        retriever_config["embedding"] = EmbeddingConfig(**retriever_config["embedding"])
        retriever_config["vector_store"] = VectorStoreConfig(**retriever_config["vector_store"])
        merged["retriever"] = DenseRetrieverConfig(**retriever_config)
        
    elif retriever_type == "sparse":
        # Sparse retriever requires dump_path
        if "dump_path" not in retriever_config:
            raise ValueError("Sparse retriever requires 'dump_path' configuration under retriever section")
        
        merged["retriever"] = SparseRetrieverConfig(**retriever_config)
        
    elif retriever_type == "ensemble":
        # Ensemble retriever requires retrievers list
        if "retrievers" not in retriever_config:
            raise ValueError("Ensemble retriever requires 'retrievers' configuration under retriever section")
        
        # Validate and process each retriever in the ensemble
        ensemble_retrievers = []
        for i, r_config in enumerate(retriever_config["retrievers"]):
            r_type = r_config.get("type")
            if r_type == "dense":
                if "embedding" not in r_config or "vector_store" not in r_config:
                    raise ValueError(f"Ensemble retriever #{i+1} (dense) requires 'embedding' and 'vector_store' configuration")
                r_config["embedding"] = EmbeddingConfig(**r_config["embedding"])
                r_config["vector_store"] = VectorStoreConfig(**r_config["vector_store"])
            elif r_type == "sparse":
                if "dump_path" not in r_config:
                    raise ValueError(f"Ensemble retriever #{i+1} (sparse) requires 'dump_path' configuration")
            elif "weight" not in r_config:
                raise ValueError(f"Ensemble retriever #{i+1} requires 'weight' configuration")
            
            ensemble_retrievers.append(r_config)
        
        retriever_config["retrievers"] = ensemble_retrievers
        merged["retriever"] = EnsembleRetrieverConfig(**retriever_config)
    
    # Create and return the RAGConfig object
    cfg = RAGConfig(**merged)
    
    global GLOBAL_VERBOSE
    GLOBAL_VERBOSE = cfg.verbose

    print(
        f"[CONFIG] Loaded experiment '{cfg.wandb.run_name}' "
        f"in project '{cfg.wandb.project}' (verbose={GLOBAL_VERBOSE})."
    )
    return cfg
    