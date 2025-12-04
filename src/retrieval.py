import time
import os
import json
import sys
from typing import Any, Dict, List
from collections import defaultdict

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from config_utils import load_config, RAGConfig, vprint, DenseRetrieverConfig, SparseRetrieverConfig


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents into a context string."""
    parts = []
    for i, d in enumerate(docs, start=1):
        snippet = d.page_content.strip()
        parts.append(f"[DOC {i} | {d.metadata.get('source', 'unknown')}]\n{snippet}")
    return "\n\n".join(parts)


def build_embeddings(retriever_cfg: DenseRetrieverConfig):
    """Build embedding model for dense retrieval."""
    if isinstance(retriever_cfg, dict):
        emb_cfg = retriever_cfg["embedding"]
    else:
        emb_cfg = retriever_cfg.embedding
    print(
        f"[RET] Loading embeddings model='{emb_cfg.model_name}' "
        f"on device='{emb_cfg.device}'..."
    )
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg.model_name,
        model_kwargs={"device": emb_cfg.device},
    )
    t1 = time.perf_counter()
    print(f"[RET] Embeddings model loaded in {t1 - t0:.3f}s.")
    return embeddings


def load_vector_store(embeddings, retriever_cfg: DenseRetrieverConfig) -> Milvus:
    """Load Milvus vector store for dense retrieval."""
    if isinstance(retriever_cfg, dict):
        vs_cfg = retriever_cfg["vector_store"]
    else:
        vs_cfg = retriever_cfg.vector_store
    uri = getattr(vs_cfg, "uri", None)
    collection_name = os.path.basename(os.path.dirname(uri).rstrip("/"))
    print(f"[RET] Loading Milvus collection '{collection_name}' from uri='{uri}'...")
    t0 = time.perf_counter()
    vectordb = Milvus(
        embeddings,
        connection_args={
                "uri": uri,
            },
        collection_name=collection_name,
    )
    t1 = time.perf_counter()
    print(f"[RET] Milvus collection '{collection_name}' loaded in {t1 - t0:.3f}s.")
    return vectordb


def get_sparse_retriever(retriever_cfg: SparseRetrieverConfig):
    """Build BM25 sparse retriever from document dump."""
    if isinstance(retriever_cfg, dict):
        top_k = retriever_cfg["top_k"]
        dump_path = retriever_cfg["dump_path"]
    else:
        top_k = retriever_cfg.top_k
        dump_path = retriever_cfg.dump_path
    chunks = [json.loads(line) for line in open(dump_path, "r") if line.strip()]
    retriever = BM25Retriever.from_documents(       
        [Document(page_content=chunk['page_content'], metadata=chunk['metadata']) for chunk in chunks],
        k=int(top_k),
    )
    return retriever


def get_dense_retriever(retriever_cfg: DenseRetrieverConfig):
    """Build dense retriever using embeddings and vector store."""
    if isinstance(retriever_cfg, dict):
        top_k = retriever_cfg["top_k"]
    else:
        top_k = retriever_cfg.top_k
    embeddings = build_embeddings(retriever_cfg)
    vectordb = load_vector_store(embeddings, retriever_cfg)
    retriever = vectordb.as_retriever(
        search_kwargs={"k": int(top_k)}
    )
    return retriever


def build_retriever(cfg: RAGConfig):
    """Build the appropriate retriever based on configuration."""
    if cfg.retriever.type == "dense":
        retriever = get_dense_retriever(cfg.retriever)
    elif cfg.retriever.type == "sparse":
        retriever = get_sparse_retriever(cfg.retriever)
    elif cfg.retriever.type == "ensemble":
        retrievers, weights = [], []
        for retriever in cfg.retriever.retrievers:
            if retriever['type'] == "dense":
                retriever = get_dense_retriever(retriever)
            elif retriever['type'] == "sparse":
                retriever = get_sparse_retriever(retriever)
            retrievers.append(retriever)
        retriever = EnsembleRetriever(retrievers=retrievers, weights=cfg.retriever.weights)
    else:
        raise ValueError(f"Unknown retriever type: {cfg.retriever.type}")
    return retriever


def retrieve_documents(question: str, retriever, cfg: RAGConfig) -> Dict[str, Any]:
    """Retrieve documents for a given question."""
    vprint(f"[RET] Retrieving docs for question: '{question[:80]}...'")
    t0 = time.perf_counter()
    
    retrieved_docs = retriever.invoke(question)
    t1 = time.perf_counter()
    
    vprint(
        f"[RET] Retrieved {len(retrieved_docs)} docs "
        f"in {t1 - t0:.3f}s (top_k={cfg.retriever.top_k})."
    )

    context_str = format_context(retrieved_docs)
    
    return {
        "question": question,
        "retrieved_docs": retrieved_docs,
        "context": context_str,
        "latency_retrieval": t1 - t0,
    }


def save_retrieval_results(results: List[Dict[str, Any]], dump_path: str):
    """Save retrieval results to a JSONL file."""
    print(f"[RET] Saving {len(results)} retrieval results to '{dump_path}'...")
    t0 = time.perf_counter()
    
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    
    with open(dump_path, "w", encoding="utf-8") as f:
        for result in results:
            # Convert Document objects to serializable format
            serializable_result = result.copy()
            serializable_result["retrieved_docs"] = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["retrieved_docs"]
            ]
            f.write(json.dumps(serializable_result, ensure_ascii=False) + "\n")
    
    t1 = time.perf_counter()
    print(f"[RET] Saved retrieval results in {t1 - t0:.3f}s.")


def load_retrieval_results(dump_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load retrieval results from a JSONL file."""
    print(f"[RET] Loading retrieval results from '{dump_path}'...")
    t0 = time.perf_counter()
    
    results_by_question = {}
    with open(dump_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            question = result["question"]
            results_by_question[question] = result
    
    t1 = time.perf_counter()
    print(f"[RET] Loaded {len(results_by_question)} retrieval results in {t1 - t0:.3f}s.")
    return results_by_question


def run_retrieval_phase(cfg: RAGConfig, questions: List[str]) -> Dict[str, Any]:
    """Run the retrieval phase for all questions and save results."""
    overall_start = time.perf_counter()
    
    # Build retriever
    retriever = build_retriever(cfg)
    infra_elapsed = time.perf_counter() - overall_start
    print(f"[RET] Retriever initialization completed in {infra_elapsed:.3f}s.")
    
    # Run retrieval for all questions
    results = []
    t_retrieval_start = time.perf_counter()
    
    for i, question in enumerate(questions):
        vprint(f"[RET] Processing question {i+1}/{len(questions)}: '{question[:80]}...'")
        result = retrieve_documents(question, retriever, cfg)
        results.append(result)
    
    t_retrieval_end = time.perf_counter()
    total_elapsed = t_retrieval_end - overall_start
    
    print(f"[RET] Retrieval phase completed:")
    print(f"[RET] - Questions processed: {len(questions)}")
    print(f"[RET] - Total time: {total_elapsed:.3f}s")
    print(f"[RET] - Avg time per question: {total_elapsed / len(questions):.3f}s")
    
    return {
        "results": results,
        "retriever": retriever,
        "total_time": total_elapsed,
        "avg_time_per_question": total_elapsed / len(questions)
    }


def main(cfg: RAGConfig):
    """Main function for retrieval phase."""
    # Load test dataset to get questions
    from generation import load_test_dataset, apply_preprocessing
    
    raw_examples = load_test_dataset(cfg)
    eval_examples = apply_preprocessing(raw_examples, cfg)
    if not eval_examples:
        print("[RET] No evaluation examples after preprocessing. Exiting.")
        return
    
    questions = [ex["question"] for ex in eval_examples]
    
    # Run retrieval phase
    retrieval_results = run_retrieval_phase(cfg, questions)
    
    # Save results if dump path is configured
    if cfg.local_dump.retrieval_dump_path:
        save_retrieval_results(retrieval_results["results"], cfg.local_dump.retrieval_dump_path)
    else:
        print("[RET] No retrieval_dump_path configured in local_dump. Results not saved.")
    
    return retrieval_results


if __name__ == "__main__":
    import argparse
    import config_utils
    from config_utils import load_config

    parser = argparse.ArgumentParser(
        description="RAG retrieval pipeline"
    )

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
        print("[RET] Verbose mode enabled via CLI.")

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
