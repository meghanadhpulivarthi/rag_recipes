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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from config_utils import load_config, RAGConfig, vprint, GLOBAL_VERBOSE

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


def build_vector_store(chunks: List[Document], cfg: RAGConfig) -> Milvus:
    emb_cfg = cfg.embedding
    vs_cfg = cfg.vector_store

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


def main(cfg: RAGConfig):
    overall_start = time.perf_counter()


    # 2) Chunk documents
    if cfg.chunking.strategy.lower() == "custom":
        chunks = custom_chunk_documents(cfg)
    else:
        docs = load_raw_documents(cfg)
        chunks = default_chunk_documents(docs, cfg)

    # 3) Build vector store
    vectordb = build_vector_store(chunks, cfg)

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

    # hyperparam overrides
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--chunk_overlap", type=int, default=None)
    parser.add_argument("--chunk_strategy", type=str, default=None)
    parser.add_argument("--embedding_model_name", type=str, default=None)
    parser.add_argument("--embedding_device", type=str, default=None)
    parser.add_argument("--uri", type=str, default=None)
    parser.add_argument("--drop_old", type=bool, default=None)

    args = parser.parse_args()

    if args.verbose:
        config_utils.GLOBAL_VERBOSE = True
        print("[INDEX] Verbose mode enabled via CLI.")

    cfg = load_config(
        default_path=args.default_config,
        task_path=args.task_config,
    )

    # CLI overrides (highest priority)
    if args.chunk_size is not None:
        cfg.chunking.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        cfg.chunking.chunk_overlap = args.chunk_overlap
    if args.chunk_strategy is not None:
        cfg.chunking.strategy = args.chunk_strategy
    if args.embedding_model_name is not None:
        cfg.embedding.model_name = args.embedding_model_name
    if args.embedding_device is not None:
        cfg.embedding.device = args.embedding_device
    if args.uri is not None:
        cfg.vector_store.uri = args.uri
    if args.drop_old is not None:
        cfg.vector_store.drop_old = args.drop_old

    main(cfg)