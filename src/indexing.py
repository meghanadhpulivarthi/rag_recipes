import os
import sys
import glob
import time
import importlib
from typing import List

# Add project root to Python path to enable imports of tasks module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.documents import Document
from langchain_milvus import Milvus
from config_utils import load_config, RAGConfig, vprint, GLOBAL_VERBOSE, SparseRetrieverConfig, DenseRetrieverConfig

def load_raw_documents(cfg: RAGConfig) -> List[Document]:
    data_cfg = cfg.data
    pattern = os.path.join(data_cfg.input_dir, data_cfg.glob_pattern)
    print(f"[INDEX] Loading raw documents from pattern: {pattern}")

    start = time.perf_counter()
    file_paths = glob.glob(pattern, recursive=True)
    docs: List[Document] = []

    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
            if not text.strip():
                continue
            docs.append(Document(page_content=text, metadata={"source": fp}))
        except Exception as e:
            print(f"[INDEX][WARN] Failed to read {fp}: {e}")

    elapsed = time.perf_counter() - start
    print(f"[INDEX] Scanned {len(file_paths)} files.")
    print(f"[INDEX] Loaded {len(docs)} documents in {elapsed:.3f}s.")
    return docs


def default_chunk_documents(docs: List[Document], cfg: RAGConfig) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_cfg = cfg.chunking
    print(
        f"[INDEX] Using default chunking: chunk_size={chunk_cfg.chunk_size}, "
        f"overlap={chunk_cfg.chunk_overlap}"
    )
    start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.chunk_size,
        chunk_overlap=chunk_cfg.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    elapsed = time.perf_counter() - start
    print(f"[INDEX] Split into {len(chunks)} chunks in {elapsed:.3f}s.")
    return chunks


def custom_chunk_documents(cfg: RAGConfig) -> List[Document]:
    chunk_cfg = cfg.chunking
    if not chunk_cfg.custom_chunker_module or not chunk_cfg.custom_chunker_fn:
        raise ValueError("[INDEX] Custom chunking requested but module/fn not configured.")

    print(
        f"[INDEX] Using custom chunker "
        f"{chunk_cfg.custom_chunker_module}.{chunk_cfg.custom_chunker_fn}"
    )
    start = time.perf_counter()
    module = importlib.import_module(chunk_cfg.custom_chunker_module)
    fn = getattr(module, chunk_cfg.custom_chunker_fn)
    params = chunk_cfg.params or {}
    chunks = fn(cfg, **params)
    elapsed = time.perf_counter() - start
    print(f"[INDEX] Custom chunker produced {len(chunks)} chunks in {elapsed:.3f}s.")
    return chunks


def build_vector_store(chunks: List[Document], retriever_cfg: DenseRetrieverConfig) -> Milvus:
    from langchain_huggingface import HuggingFaceEmbeddings

    if isinstance(retriever_cfg, dict):
        emb_cfg = retriever_cfg["embedding"]
        vs_cfg = retriever_cfg["vector_store"]
    else:
        emb_cfg = retriever_cfg.embedding
        vs_cfg = retriever_cfg.vector_store

    print(f"[INDEX] Building embeddings with model='{emb_cfg.model_name}', device='{emb_cfg.device}'")
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg.model_name,
        model_kwargs={"device": emb_cfg.device},
    )
    t1 = time.perf_counter()
    print(f"[INDEX] Embedding model loaded in {t1 - t0:.3f}s.")

    # Derive a Milvus collection name; fall back to a sensible default
    uri = getattr(vs_cfg, "uri", None)
    collection_name = os.path.basename(os.path.dirname(uri).rstrip("/"))

    os.makedirs(os.path.dirname(uri), exist_ok=True)
    print(f"[INDEX] Building Milvus collection '{collection_name}' from {len(chunks)} chunks...")
    t2 = time.perf_counter()
    vectordb = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={
            "uri": uri,
        },
        collection_name=collection_name,
        drop_old=vs_cfg.drop_old,
    )
    t3 = time.perf_counter()
    print(f"[INDEX] Milvus collection '{collection_name}' built in {t3 - t2:.3f}s.")
    return vectordb

def dump_chunks(chunks: List[Document], retriever_cfg: SparseRetrieverConfig):
    import json
    if isinstance(retriever_cfg, dict):
        dump_path = retriever_cfg["dump_path"]
    else:
        dump_path = retriever_cfg.dump_path
    with open(dump_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Convert Document to dict for JSON serialization
            chunk_dict = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")

def main(cfg: RAGConfig):
    overall_start = time.perf_counter()


    # 2) Chunk documents
    if cfg.chunking.strategy.lower() == "custom":
        chunks = custom_chunk_documents(cfg)
    else:
        docs = load_raw_documents(cfg)
        chunks = default_chunk_documents(docs, cfg)

    # 3) Build vector store
    if cfg.retriever.type == "sparse":
        dump_chunks(chunks, cfg.retriever)
    elif cfg.retriever.type == "dense":
        vectordb = build_vector_store(chunks, cfg.retriever)
    elif cfg.retriever.type == "ensemble":
        for retriever in cfg.retriever.retrievers:
            if retriever["type"] == "dense":
                vectordb = build_vector_store(chunks, retriever)
            elif retriever["type"] == "sparse":
                dump_chunks(chunks, retriever)
    overall_elapsed = time.perf_counter() - overall_start
    print(f"[INDEX] Full indexing pipeline completed in {overall_elapsed:.3f}s.")


if __name__ == "__main__":
    import argparse
    import config_utils
    from config_utils import load_config

    parser = argparse.ArgumentParser(description="RAG indexing pipeline")

    parser.add_argument("--default_config", type=str, default="configs/default.yaml")
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    # Parse known args first to get task_config path
    known_args, _ = parser.parse_known_args()
    
    # Add dynamic CLI arguments based on defaults in task config
    task_config_path = known_args.task_config or "configs/ingestbench_dense.yaml"
    arg_mapping = config_utils.add_dynamic_cli_args(parser, task_config_path)

    args = parser.parse_args()

    if args.verbose:
        config_utils.GLOBAL_VERBOSE = True
        print("[INDEX] Verbose mode enabled via CLI.")

    # Extract dynamic CLI overrides
    cli_overrides = {}
    for cli_arg, var_name in arg_mapping.items():
        cli_value = getattr(args, var_name, None)
        if cli_value is not None:
            cli_overrides[var_name] = cli_value
            print(f"[CONFIG] CLI override: {var_name} = {cli_value}")

    cfg = load_config(
        default_path=args.default_config,
        task_path=args.task_config,
        cli_overrides=cli_overrides,
    )

    main(cfg)