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
