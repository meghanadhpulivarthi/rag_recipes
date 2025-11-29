import time
import os, json
import csv
import importlib
import sys
from collections import defaultdict
from typing import Any, Dict, List
import torch.distributed as dist

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import wandb
from jinja2 import Template
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config_utils import load_config, RAGConfig, vprint
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
from IPython.core.debugger import Pdb
from datetime import datetime
import uuid



# ------------- Dataset loading -------------


def load_test_dataset(cfg: RAGConfig) -> List[Dict[str, Any]]:
    td_cfg = cfg.test_data
    path = td_cfg.path
    loader = td_cfg.loader.lower()

    print(f"[GEN] Loading test dataset from '{path}' with loader='{loader}'...")
    t0 = time.perf_counter()

    if loader == "jsonl":
        examples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    elif loader == "csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            examples = [row for row in reader]
    else:
        raise ValueError(f"[GEN] Unsupported loader type: {loader}")

    elapsed = time.perf_counter() - t0
    print(f"[GEN] Loaded {len(examples)} raw test examples in {elapsed:.3f}s.")
    return examples


def apply_preprocessing(raw_examples: List[Dict[str, Any]], cfg: RAGConfig) -> List[Dict[str, Any]]:
    pp_cfg = cfg.preprocessing
    print(
        f"[GEN] Applying preprocessing function "
        f"{pp_cfg.module}.{pp_cfg.function} to dataset..."
    )
    t0 = time.perf_counter()
    module = importlib.import_module(pp_cfg.module)
    fn = getattr(module, pp_cfg.function)
    params = pp_cfg.params or {}
    processed = fn(raw_examples, cfg, **params)
    elapsed = time.perf_counter() - t0
    print(f"[GEN] Preprocessing produced {len(processed)} (question, gold) pairs in {elapsed:.3f}s.")
    return processed

def load_schema(schema_path: str):
    # schema_path like "tasks.ingestbench.custom_functions.output_schema"
    module_path, class_name = schema_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    schema_cls = getattr(module, class_name)
    return schema_cls

# ------------- Embeddings, index, LLM -------------


def build_embeddings(cfg: RAGConfig):
    emb_cfg = cfg.embedding
    print(
        f"[GEN] Loading embeddings model='{emb_cfg.model_name}' "
        f"on device='{emb_cfg.device}'..."
    )
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg.model_name,
        model_kwargs={"device": emb_cfg.device},
    )
    t1 = time.perf_counter()
    print(f"[GEN] Embeddings model loaded in {t1 - t0:.3f}s.")
    return embeddings


def load_vector_store(embeddings, cfg: RAGConfig) -> Milvus:
    vs_cfg = cfg.vector_store

    uri = getattr(vs_cfg, "uri", None)
    collection_name = os.path.basename(os.path.dirname(uri).rstrip("/"))
    print(f"[GEN] Loading Milvus collection '{collection_name}' from uri='{uri}'...")
    t0 = time.perf_counter()
    vectordb = Milvus(
        embeddings,
        connection_args={
                "uri": uri,
            },
        collection_name=collection_name,
    )
    t1 = time.perf_counter()
    print(f"[GEN] Milvus collection '{collection_name}' loaded in {t1 - t0:.3f}s.")
    return vectordb


def build_llm(cfg: RAGConfig):
    llm_cfg = cfg.llm
    provider = llm_cfg.provider.lower()

    # ------ OpenAI path ------
    if provider == "openai":
        vprint(
            f"[GEN] Initializing OpenAI LLM model='{llm_cfg.model_name}', "
            f"temperature={llm_cfg.temperature}, max_tokens={llm_cfg.max_tokens}"
        )
        t0 = time.perf_counter()
        llm = ChatOpenAI(
            model=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            response_format=llm_cfg.response_format,
        )
        t1 = time.perf_counter()
        print(f"[GEN] OpenAI LLM initialized in {t1 - t0:.3f}s.")
        return llm

    # ------ HF pipeline path ------
    if provider in {"hf_pipeline", "local_hf", "hf"}:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import HuggingFacePipeline
        task = getattr(llm_cfg, "pipeline_task", "text-generation")
        model_id = llm_cfg.model_name

        vprint(
            f"[GEN] Initializing HF pipeline "
            f"task='{task}', model_id='{model_id}', "
            f"temperature={llm_cfg.temperature}, max_new_tokens={llm_cfg.max_tokens}"
        )

        t0 = time.perf_counter()
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=llm_cfg.max_tokens,
            do_sample=(llm_cfg.temperature > 0),
            temperature=llm_cfg.temperature if llm_cfg.temperature > 0 else 1.0,
            return_full_text=False,  # important: only continuation
            device=llm_cfg.device,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        t1 = time.perf_counter()
        print(f"[GEN] HF pipeline loaded on device={device} in {t1 - t0:.3f}s.")

        # just return the pipeline object
        return llm
    
    if provider in {"vllm", "local_vllm"}:
        from vllm import LLM

        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        t0 = time.perf_counter()
        llm = LLM(model=llm_cfg.model_name, seed=llm_cfg.seed)
        t1 = time.perf_counter()
        print(f"[GEN] vLLM LLM initialized on device={llm_cfg.device} in {t1 - t0:.3f}s.")
        return llm

    raise ValueError(f"[GEN] LLM provider '{llm_cfg.provider}' not implemented.")


# ------------- RAG core -------------


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        snippet = d.page_content.strip()
        parts.append(f"[DOC {i} | {d.metadata.get('source', 'unknown')}]\n{snippet}")
    return "\n\n".join(parts)


def run_rag_query(
    question: str,
    vectordb: Milvus,
    llm: ChatOpenAI,
    cfg: RAGConfig,
) -> Dict[str, Any]:
    vprint(f"[GEN] Running RAG query for question: '{question[:80]}...'")
    t0 = time.perf_counter()

    # Retrieval
    retriever = vectordb.as_retriever(
        search_kwargs={"k": cfg.retriever.top_k}
    )
    t_retr_start = time.perf_counter()
    retrieved_docs = retriever.invoke(question)
    t_retr_end = time.perf_counter()
    vprint(
        f"[GEN] Retrieved {len(retrieved_docs)} docs "
        f"in {t_retr_end - t_retr_start:.3f}s (top_k={cfg.retriever.top_k})."
    )

    # Context + prompt from config template
    context_str = format_context(retrieved_docs)
    prompt_cfg = cfg.prompt
    template = Template(prompt_cfg.template)
    rendered_prompt = template.render(
        system_message=prompt_cfg.system_message,
        context=context_str,
        question=question,
    )
    vprint("[GEN] Rendered prompt:")
    vprint(rendered_prompt)

    # LLM call
    t_llm_start = time.perf_counter()
    if cfg.llm.provider in {"hf", "hf_pipeline", "local_hf", "openai"}:
        resp = llm.invoke(rendered_prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    elif cfg.llm.provider in {"vllm", "local_vllm"}:
        # vLLM path
        if hasattr(cfg.llm, 'response_format') and cfg.llm.response_format is not None:
            schema = load_schema(cfg.llm.response_format)
        else:
            schema = None
        
        # Format messages for vLLM
        messages = [{"role": "user", "content": rendered_prompt}]
        
        guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema() if schema else None)
        sampling_params = SamplingParams(
            guided_decoding=guided_decoding_params,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        resp = llm.chat(messages, sampling_params=sampling_params)
        answer = resp[0].outputs[0].text
    t_llm_end = time.perf_counter()
    vprint(f"[GEN] LLM call completed in {t_llm_end - t_llm_start:.3f}s.")

    t1 = time.perf_counter()
    vprint(f"[GEN] Full RAG query pipeline completed in {t1 - t0:.3f}s.")

    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "context": context_str,
        "prompt": rendered_prompt,
        "latency_total": t1 - t0,
        "latency_retrieval": t_retr_end - t_retr_start,
        "latency_llm": t_llm_end - t_llm_start,
    }



# -------- Metrics + W&B --------

def init_wandb(cfg: RAGConfig):
    if not cfg.wandb.enabled:
        print("[GEN] W&B disabled in config.")
        return None

    print(f"[GEN] Initializing W&B run: project={cfg.wandb.project}, name={cfg.wandb.run_name}")
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config={
            "verbose": cfg.verbose,
            "chunking": vars(cfg.chunking),
            "embedding": vars(cfg.embedding),
            "retriever": vars(cfg.retriever),
            "vector_store": vars(cfg.vector_store),
            "llm": vars(cfg.llm),
            "test_data": vars(cfg.test_data),
            "metrics": vars(cfg.metrics),
        },
    )
    return run


def load_metric_fn(cfg: RAGConfig):
    m_cfg = cfg.metrics
    print(f"[GEN] Loading metric function {m_cfg.module}.{m_cfg.function} ...")
    module = importlib.import_module(m_cfg.module)
    fn = getattr(module, m_cfg.function)
    return fn

def main(cfg: RAGConfig):
    overall_start = time.perf_counter()

    # W&B + metric function
    run = init_wandb(cfg)
    if run:
        print(f"[GEN] W&B run initialized: {run.name}")
    metric_fn = load_metric_fn(cfg)

    # Infra
    embeddings = build_embeddings(cfg)
    vectordb = load_vector_store(embeddings, cfg)
    llm = build_llm(cfg)

    infra_elapsed = time.perf_counter() - overall_start
    print(f"[GEN] Initialization (embeddings + index + LLM) completed in {infra_elapsed:.3f}s.")

    # Dataset
    raw_examples = load_test_dataset(cfg)
    eval_examples = apply_preprocessing(raw_examples, cfg)
    if not eval_examples:
        print("[GEN] No evaluation examples after preprocessing. Exiting.")
        return

    print(f"[GEN] Starting RAG evaluation on {len(eval_examples)} examples...")
    t_eval_start = time.perf_counter()

    # Metric aggregation + W&B table
    metric_sums = defaultdict(float)
    metric_keys: List[str] = []
    wb_table = None

    dump_cfg = cfg.local_dump

    if dump_cfg.enabled:
        out_dir = dump_cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base, ext = os.path.splitext(dump_cfg.filename)
        filename_with_ts = f"{base}_{timestamp}{ext}"
        dump_path = os.path.join(out_dir, filename_with_ts)

    payloads = []
    for i, ex in enumerate(eval_examples):
        q = ex["question"]
        gold = ex["gold"]
        meta = ex.get("meta", {})

        result = run_rag_query(q, vectordb, llm, cfg)
        pred = result["answer"].strip()
        gold_stripped = gold.strip()

        # Compute custom metrics
        metric_params = cfg.metrics.params or {}
        metric_dict = metric_fn(pred, gold_stripped, meta, cfg, result, **metric_params)
        if not metric_keys:
            metric_keys = sorted(metric_dict.keys())
            # initialize table once we know the metric names
            wb_table = wandb.Table(
                columns=["idx", "question", "meta", "prediction", "gold", "prompt", "retrieved_docs", "num_retrieved", "latency_total", "latency_retrieval", "latency_llm"] + [f"metrics/{k}" for k in metric_keys]
            ) if cfg.wandb.enabled and wandb.run is not None else None

        for k, v in metric_dict.items():
            metric_sums[k] += float(v)

        # Example-level prints
        vprint(f"\n[EXAMPLE {i}]")
        vprint(f"Q: {q}")
        vprint(f"PRED: {pred}")
        vprint(f"GOLD: {gold}")
        for k in metric_keys:
            vprint(f"{k}: {metric_dict.get(k)}")

        # Per-example logging to W&B
        docs_short = []
        for d in result["retrieved_docs"]:
            docs_short.append({
                "source": d.metadata.get("source", "unknown"),
                "text": d.page_content[:500],
            })

        log_payload = {
            "idx": i,
            "question": q,
            "meta": meta,
            "prediction": pred,
            "gold": gold_stripped,
            "prompt": result["prompt"],
            "retrieved_docs": docs_short,
            "num_retrieved": len(result["retrieved_docs"]),
            "latency_total": result["latency_total"],
            "latency_retrieval": result["latency_retrieval"],
            "latency_llm": result["latency_llm"],
        }

        # add metric fields, e.g. metrics/exact_match, metrics/token_f1
        for k, v in metric_dict.items():
            log_payload[f"metrics/{k}"] = float(v)

        payloads.append(log_payload)
        if dump_cfg.enabled:
            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_payload, ensure_ascii=False) + "\n")
        if cfg.wandb.enabled and wandb.run is not None:
            wb_table.add_data(*[log_payload[col] for col in wb_table.columns])

    # Aggregate metrics
    t_eval_end = time.perf_counter()
    total_elapsed = t_eval_end - t_eval_start
    n = len(eval_examples)

    print("\n[GEN] Evaluation completed.")
    print(f"[GEN] Examples: {n}")
    for k in metric_keys:
        mean_val = metric_sums[k] / n
        print(f"[GEN] Mean {k}: {mean_val:.4f}")
    print(f"[GEN] Total eval time: {total_elapsed:.3f}s ")
    print(f"[GEN] Avg time per example: ({total_elapsed / n:.3f}s/example)")

    if cfg.wandb.enabled and wandb.run is not None:
        # log aggregate metrics
        agg_payload = {
            "eval_num_examples": n,
            "eval_total_time_sec": total_elapsed,
        }
        for k in metric_keys:
            agg_payload[f"eval_mean_{k}"] = metric_sums[k] / n

        wandb.log(agg_payload)

        # log the table itself
        if wb_table is not None:
            wandb.log({"eval_table": wb_table})

        if cfg.wandb.custom_logging_function:
            module_path, function_name = cfg.wandb.custom_logging_function.rsplit('.', 1)
            custom_module = importlib.import_module(module_path)
            custom_logging_fn = getattr(custom_module, function_name)
            custom_logging_fn(cfg, wandb, payloads)

        wandb.finish()
        print("[GEN] W&B run finished.")

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("Destroyed torch distributed process group.")
    except Exception as e:
        print("Exception when destroying process group:", e)



if __name__ == "__main__":
    import argparse
    import config_utils
    from config_utils import load_config

    parser = argparse.ArgumentParser(
        description="RAG generation/eval pipeline with custom metrics"
    )

    parser.add_argument("--default_config", type=str, default="configs/default.yaml")
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    # wandb
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # retrieval
    parser.add_argument("--top_k", type=int, default=None)

    # LLM
    parser.add_argument("--llm_provider", type=str, default=None)
    parser.add_argument("--llm_model_name", type=str, default=None)
    parser.add_argument("--llm_temperature", type=float, default=None)
    parser.add_argument("--llm_max_tokens", type=int, default=None)
    parser.add_argument("--llm_pipeline_task", type=str, default=None)


    # test data
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_data_loader", type=str, default=None)

    # preprocessing / metrics
    parser.add_argument("--preproc_module", type=str, default=None)
    parser.add_argument("--preproc_fn", type=str, default=None)
    parser.add_argument("--metrics_module", type=str, default=None)
    parser.add_argument("--metrics_fn", type=str, default=None)

    args = parser.parse_args()

    if args.verbose:
        config_utils.GLOBAL_VERBOSE = True
        print("[GEN] Verbose mode enabled via CLI.")

    cfg = load_config(
        default_path=args.default_config,
        task_path=args.task_config,
    )

    # CLI overrides
    if args.wandb_enabled is not None:
        cfg.wandb.enabled = args.wandb_enabled
    if args.wandb_project is not None:
        cfg.wandb.project = args.wandb_project
    if args.wandb_run_name is not None:
        cfg.wandb.run_name = args.wandb_run_name

    if args.top_k is not None:
        cfg.retriever.top_k = args.top_k

    if args.llm_provider is not None:
        cfg.llm.provider = args.llm_provider
    if args.llm_model_name is not None:
        cfg.llm.model_name = args.llm_model_name
    if args.llm_temperature is not None:
        cfg.llm.temperature = args.llm_temperature
    if args.llm_max_tokens is not None:
        cfg.llm.max_tokens = args.llm_max_tokens
    if args.llm_pipeline_task is not None:
        cfg.llm.pipeline_task = args.llm_pipeline_task

    if args.test_data_path is not None:
        cfg.test_data.path = args.test_data_path
    if args.test_data_loader is not None:
        cfg.test_data.loader = args.test_data_loader

    if args.preproc_module is not None:
        cfg.preprocessing.module = args.preproc_module
    if args.preproc_fn is not None:
        cfg.preprocessing.function = args.preproc_fn
    if args.metrics_module is not None:
        cfg.metrics.module = args.metrics_module
    if args.metrics_fn is not None:
        cfg.metrics.function = args.metrics_fn

    main(cfg)