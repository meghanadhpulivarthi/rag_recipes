# RAG Recipes - Package Guide

**RAG Recipes** is a simple, modular package for building Retrieval-Augmented Generation (RAG) pipelines. It provides reusable components for indexing documents, retrieving relevant content, and generating responses with language models.

## Key Features

- Multiple retrieval strategies (dense, sparse, hybrid)
- YAML-based configuration for all pipeline parameters
- CLI tools for indexing, retrieval, and generation
- Support for multiple LLM providers and backends
- Custom preprocessing and metrics via project-specific functions

## Core Components

RAG Recipes consists of four main modules:

### Configuration Management
Handles loading, parsing, and managing YAML configuration files. Supports hierarchical configurations with variable substitution and environment overrides. All parameters for indexing, retrieval, and generation are managed through this interface.

### Document Indexing
Processes raw documents into manageable chunks suitable for retrieval. Provides multiple chunking strategies (fixed-size, semantic, sliding-window) and supports custom preprocessing pipelines. Integrates with vector databases for embeddings.

### Retrieval
Implements multiple retrieval backends including dense vector similarity (embedding-based), sparse keyword matching (BM25), and hybrid approaches. Each strategy is independently configurable and can be swapped via configuration.

### Generation
Wraps language model inference with templated prompt generation, context integration, and output formatting. Supports both open-source models via Hugging Face and commercial providers. Handles model loading, batching, and response post-processing.

## Package Architecture

```
rag_recipes/
├── src/                    # Core package modules
│   ├── config_utils.py     # Configuration management
│   ├── indexing.py         # Document processing and chunking
│   ├── retrieval.py        # Retrieval backends
│   ├── generation.py       # LLM inference and generation
│   ├── cli.py              # Command-line interface
│   └── utils.py            # Shared utilities
├── configs/                # Example configurations (reference only)
├── tasks/                  # Sample task implementations
├── pyproject.toml          # Package metadata and dependencies
└── PACKAGE_GUIDE.md        # This file
```

## Installation

Install from GitHub:

```bash
uv add git+ssh://git@github.ibm.com:Meghanadh-Pulivarthi1/rag_recipes.git
```

Or with pip:

```bash
pip install git+ssh://git@github.ibm.com:Meghanadh-Pulivarthi1/rag_recipes.git
```

## Getting Started

### Basic Workflow

1. **Install** RAG Recipes in your project environment
2. **Create a configuration file** (YAML) describing your indexing, retrieval, and generation strategy
3. **Organize your data** in a directory structure accessible from your project
4. **Run indexing** to prepare documents for retrieval
5. **Run retrieval** to search over indexed documents
6. **Run generation** to produce answers using retrieved context

All three phases are available as CLI commands that can be invoked from any directory:

- `rag-index`: Process and chunk documents, create vector indices
- `rag-retrieve`: Search indexed documents for relevant context
- `rag-generate`: Generate responses with retrieved context using an LLM

### Command-Line Interface

All CLI commands support the following standard arguments plus dynamic configuration overrides.

#### Common Arguments (All Commands)

All three CLI commands (`rag-index`, `rag-retrieve`, `rag-generate`) support these base arguments:

- `--default_config <path>` - Path to the default configuration file (default: `configs/default.yaml`)
- `--task_config <path>` - Path to the task-specific configuration file that overrides defaults (optional)
- `--verbose` - Enable verbose logging output (flag, no value needed)

#### Dynamic Configuration Overrides

In addition to the common arguments, RAG Recipes supports **dynamic CLI arguments** that are automatically generated from your configuration file's `defaults` section. Any key in the `defaults` section of your YAML configuration can be overridden via CLI using the format:

```bash
rag-index --key_name value
```

For example, if your config has:
```yaml
defaults:
  chunk_size: 512
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

You can override these at runtime:
```bash
rag-index --chunk_size 1024 --model_name "sentence-transformers/all-mpnet-base-v2"
```

#### rag-index Command

**Purpose**: Process and chunk raw documents, build vector indices for dense retrieval, or dump chunks for sparse retrieval.

**Basic Usage**:
```bash
rag-index --default_config configs/default.yaml --task_config configs/my_task.yaml
```

**Arguments**:
- `--default_config <path>` - Default configuration file (default: `configs/default.yaml`)
- `--task_config <path>` - Task-specific config that overrides defaults (optional)
- `--verbose` - Enable verbose logging
- `--<config_var>` - Any variable from `defaults` section in config file

**Example**:
```bash
rag-index \
  --default_config configs/default.yaml \
  --task_config configs/ingestbench_dense.yaml \
  --verbose \
  --chunk_size 512 \
  --chunk_overlap 50
```

#### rag-retrieve Command

**Purpose**: Search indexed documents and retrieve relevant context for given questions.

**Basic Usage**:
```bash
rag-retrieve --default_config configs/default.yaml --task_config configs/my_task.yaml
```

**Arguments**:
- `--default_config <path>` - Default configuration file (default: `configs/default.yaml`)
- `--task_config <path>` - Task-specific config that overrides defaults (optional)
- `--verbose` - Enable verbose logging
- `--<config_var>` - Any variable from `defaults` section in config file

**Example**:
```bash
rag-retrieve \
  --default_config configs/default.yaml \
  --task_config configs/ingestbench_dense.yaml \
  --verbose \
  --top_k 10
```

#### rag-generate Command

**Purpose**: Run RAG evaluation pipeline - retrieve context, generate responses using LLM, compute metrics.

**Basic Usage**:
```bash
rag-generate --default_config configs/default.yaml --task_config configs/my_task.yaml
```

**Arguments**:
- `--default_config <path>` - Default configuration file (default: `configs/default.yaml`)
- `--task_config <path>` - Task-specific config that overrides defaults (optional)
- `--verbose` - Enable verbose logging
- `--<config_var>` - Any variable from `defaults` section in config file

**Example**:
```bash
rag-generate \
  --default_config configs/default.yaml \
  --task_config configs/ingestbench_dense.yaml \
  --verbose \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --temperature 0.7 \
  --max_tokens 512
```

#### Configuration-Based Arguments

The `defaults` section of your YAML configuration file can contain any of these common parameters that will be exposed as CLI arguments:

**Chunking Parameters**:
- `chunk_size` (int) - Size of text chunks in characters (e.g., 512, 1024)
- `chunk_overlap` (int) - Overlap between consecutive chunks (e.g., 50, 100)

**Retrieval Parameters**:
- `top_k` (int) - Number of documents to retrieve (e.g., 5, 10, 20)
- `embedding_model_name` (str) - Embedding model identifier (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- `embedding_device` (str) - Device for embedding model: `cpu`, `cuda`, `cuda:0`, etc.

**LLM Parameters**:
- `model_name` (str) - Language model identifier (e.g., `gpt-3.5-turbo`, `meta-llama/Llama-2-7b-hf`)
- `provider` (str) - LLM provider: `openai`, `hf`, `vllm`, `rits`, etc.
- `temperature` (float) - Sampling temperature (0.0-1.0, default: 0.7)
- `max_tokens` (int) - Maximum tokens in generation (e.g., 256, 512, 1024)
- `seed` (int) - Random seed for reproducibility (optional)

**Data & File Parameters**:
- `input_dir` (str) - Directory containing raw documents
- `glob_pattern` (str) - Glob pattern to match documents (e.g., `**/*.txt`)
- `output_dir` (str) - Directory for output results
- `filename` (str) - Output filename for results

**Examples with Different Configurations**:

Dense retrieval with Llama-2:
```bash
rag-generate \
  --task_config configs/ingestbench_dense.yaml \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --provider "hf" \
  --temperature 0.5
```

Sparse retrieval (BM25):
```bash
rag-retrieve \
  --task_config configs/ingestbench_sparse.yaml \
  --top_k 15
```

Custom chunking:
```bash
rag-index \
  --task_config configs/my_custom_task.yaml \
  --chunk_size 2048 \
  --chunk_overlap 200
```

#### Configuration Priority

When using multiple configurations and CLI arguments, the priority order is (highest to lowest):

1. **CLI arguments** (e.g., `--chunk_size 512`)
2. **Task configuration file** (specified with `--task_config`)
3. **Default configuration file** (specified with `--default_config`)

This means CLI arguments override task config, which overrides default config.

## Configuration

RAG Recipes uses YAML-based configuration files that specify all pipeline parameters: data sources, chunking strategy, retriever type, LLM model, prompt templates, and optional custom functions for preprocessing and metrics computation. Configuration files live in your project, not in RAG Recipes, allowing different projects to use RAG Recipes with completely different setups.

## Design Philosophy

RAG Recipes operates on a simple principle: the package provides generic, reusable components while your project provides configuration, data, and project-specific customizations. All three CLI tools (`rag-index`, `rag-retrieve`, `rag-generate`) work from any directory with no directory dependencies.

## Configuration Deep Dive

Configuration files control all aspects of the RAG pipeline. Key sections include:

- **verbose**: Logging level and detail
- **wandb**: Integration with Weights & Biases for experiment tracking
- **data**: Input data locations and loading patterns
- **chunking**: Strategy and parameters for document segmentation
- **retriever**: Type, model, and parameters for the retrieval phase
- **llm**: Model selection, provider, and generation parameters
- **prompt**: System messages and prompt templates with context integration
- **test_data**: Evaluation data specification
- **preprocessing** (optional): Custom preprocessing function references
- **metrics** (optional): Custom metric computation function references
- **local_dump**: Output directory and result file naming

See example configurations in `rag_recipes/configs/` for reference structures and typical values.

## Integration Guide

To integrate RAG Recipes into an existing project:

1. Add RAG Recipes as a dependency in your project's package configuration
2. Create a `configs/` directory for your RAG configurations
3. Create a `src/custom_functions.py` or similar for project-specific preprocessing and metrics
4. Write a configuration file referencing your custom functions
5. Reference this configuration when calling `rag-index`, `rag-retrieve`, and `rag-generate`

For a complete example, see the IngestBench project integration.
