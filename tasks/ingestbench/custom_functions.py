from typing import List, Dict, Any, Optional
from src.config_utils import RAGConfig, vprint
from langchain_core.documents import Document
import pandas as pd
from pydantic import BaseModel, Field
import ast, json
import numpy as np
import matplotlib.pyplot as plt
import wandb
from IPython.core.debugger import Pdb

def _normalize_list(answer: List[str]) -> List[str]:
    normalized = []
    for a in answer:
        a = a.replace("\u202f", " ")  # replace narrow no-break space with regular space
        a = a.lower().strip()
        normalized.append(a)
    return normalized

def chunker(cfg: RAGConfig, input_path: str) -> List[Document]:
    data = pd.read_json(input_path, lines=True)
    docs = []
    for _, row in data.iterrows():
        docs.append(Document(page_content=row["paraphrased_prose"], metadata={"source": row["doc_id"], "doc_id": row["doc_id"], "entity_id": row["entity_id"], "num_words": row["num_paraphrased_prose_words"]}))
    return docs

def preprocess_dataset(
    raw_examples: List[Dict[str, Any]],
    cfg: RAGConfig,
    num_examples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Turn raw dataset records into a standard format:
    [{"question": str, "gold": str, "meta": {...}}, ...]

    This is corpus-specific; adapt it to your test set schema.
    The default implementation expects JSON/CSV rows with
    fields 'question' and 'answer'.
    """
    print("[PREPROC] Starting dataset preprocessing...")
    processed = []
    if num_examples is not None:
        raw_examples = raw_examples[:num_examples]
    for i, ex in enumerate(raw_examples):
        # Example assumptions; change to what your dataset actually uses
        q = ex.get("paraphrased_question")
        gold = ex.get("answer")

        if q is None or gold is None:
            print(f"[PREPROC][WARN] Skipping example {i} due to missing fields.")
            continue

        item = {
            "question": str(q),
            "gold": str(gold),
            "meta": {
                "idx": i,
                **{k: v for k, v in ex.items() if k not in {"paraphrased_question", "answer"}},
            },
        }
        processed.append(item)

    print(f"[PREPROC] Completed preprocessing. Kept {len(processed)} examples.")
    return processed

def compute_metrics(
    prediction: str,
    gold: str,
    meta: Dict[str, Any],
    cfg: RAGConfig,
    result: Dict[str, Any],
) -> Dict[str, float]:
    """
    Custom metric function.
    Returns any number of metrics as a dict of {name: value}.

    This sample:
        - exact_match: 1.0 if normalized strings match, else 0.0
        - token_f1: simple F1 over token overlap
    """
    print("[METRICS] Computing retrieval metrics for one example...")

    retrieved_doc_ids = [d.metadata["doc_id"] for d in result["retrieved_docs"]]
    gold_doc_ids = meta["reference_doc_ids"]
    retrieved_doc_ids = set(retrieved_doc_ids)
    gold_doc_ids = set(gold_doc_ids)
    retrieval_recall = len(retrieved_doc_ids.intersection(gold_doc_ids)) / len(gold_doc_ids)
    retrieval_precision = len(retrieved_doc_ids.intersection(gold_doc_ids)) / len(retrieved_doc_ids)
    if retrieval_recall + retrieval_precision > 0:
        retrieval_f1 = 2 * retrieval_recall * retrieval_precision / (retrieval_recall + retrieval_precision)
    else:
        retrieval_f1 = 0.0

    print("[METRICS] Computing generation metrics for one example...")

    try:
        if isinstance(prediction, dict):
            prediction_dict = prediction
        else:
            prediction_dict = json.loads(prediction)
        pred_facts = list(set(prediction_dict["answer"]))
        pred_facts = _normalize_list(pred_facts)
    except Exception as e:
        pred_facts = []
    gt_facts = list(set(ast.literal_eval(gold)))
    gt_facts = _normalize_list(gt_facts)

    print(f"[METRICS] pred_facts: {pred_facts}")
    print(f"[METRICS] gt_facts: {gt_facts}")
    fact_exact_match = 1.0 if pred_facts == gt_facts else 0.0

    recalled = 0
    for gf in gt_facts:
        for pf in pred_facts:
            if gf == pf:
                recalled += 1
                break

    fact_recall = recalled / len(gt_facts)

    if len(pred_facts) == 0:
        fact_precision = 0.0
    else:
        fact_precision = recalled / len(pred_facts)
    if fact_recall + fact_precision > 0:
        fact_f1 = 2 * fact_recall * fact_precision / (fact_recall + fact_precision)
    else:
        fact_f1 = 0.0
    return {
        "retrieval_recall": retrieval_recall,
        "retrieval_precision": retrieval_precision,
        "retrieval_f1": retrieval_f1,
        "num_retrieved_docs": len(retrieved_doc_ids),
        "num_gold_docs": len(gold_doc_ids),
        "fact_recall": fact_recall,
        "fact_precision": fact_precision,
        "fact_f1": fact_f1,
        "fact_em": fact_exact_match,
        "num_fact_recalled": recalled,
        "num_gt_facts": len(gt_facts),
        "num_pred_facts": len(pred_facts)
    }


class OutputSchema(BaseModel):
    answer: List[str]


def log_depth_heatmaps(cfg: RAGConfig, wandb, eval_rows: List[Dict], ret_key: str = "retrieval_depth", rd_key: str = "reasoning_depth"):
    """
    Given a list of evaluation result dicts (each containing metadata and metrics),
    build heatmaps per metric + count-per-(ret, rd), and log them to wandb.

    eval_rows: list of dicts; each dict must have:
        - row[\"meta\"][ret_key] and row[\"meta\"][rd_key] (integers or strings like \"ret1\", \"RD2\" etc) 
        - row[\"metrics/<metric_name>\"] for each metric_name in metrics

    metrics: list of metric names (strings)
    ret_key, rd_key: keys in meta storing retrieval and reasoning depths

    Assumes ret and rd are ordinal (e.g. integers or strings convertible).
    """

    factiod_rows = [row for row in eval_rows if row['meta'].get('answer_type') == 'factoid']
    binary_rows = [row for row in eval_rows if row['meta'].get('answer_type') == 'binary']
    list_rows = [row for row in eval_rows if row['meta'].get('answer_type') == 'list']

    for rows, qtype in [(factiod_rows, 'factoid'), (binary_rows, 'binary') , (list_rows, 'list')]:
        metrics = [col.split("/")[1] for col in rows[0].keys() if col.startswith("metrics/")]
        # 1. Parse all unique values of ret / rd depths
        all_rets = sorted({row["meta"][ret_key] for row in rows})
        all_rds  = sorted({row["meta"][rd_key]  for row in rows})

        # Try to canonicalize to strings like "RET1" / "RD1", but you can customize
        ret_labels = [f"RET{r}" for r in all_rets]
        rd_labels  = [f"RD{r}"  for r in all_rds]

        # 2. Build empty dataframes: one for count, one per metric
        count_df = pd.DataFrame(0, index=rd_labels, columns=ret_labels, dtype=float)
        metric_dfs = {
            m: pd.DataFrame(np.nan, index=rd_labels, columns=ret_labels, dtype=float)
            for m in metrics
        }

        # For metrics: we will accumulate sum and count to compute mean per cell
        metric_sum = {m: { (rd, ret): 0.0 for rd in all_rds for ret in all_rets } for m in metrics}
        metric_cnt = {m: { (rd, ret): 0      for rd in all_rds for ret in all_rets } for m in metrics}

        # 3. Aggregate
        for row in rows:
            rd = row["meta"][rd_key]
            ret = row["meta"][ret_key]
            rd_label = f"RD{rd}"
            ret_label = f"RET{ret}"
            count_df.loc[rd_label, ret_label] += 1

            for m in metrics:
                val = row.get(f"metrics/{m}")
                if val is None:
                    continue
                metric_sum[m][(rd, ret)] += float(val)
                metric_cnt[m][(rd, ret)] += 1

        # 4. Fill metric_dfs with mean (or nan if no samples)
        for m in metrics:
            for rd in all_rds:
                for ret in all_rets:
                    cnt = metric_cnt[m][(rd, ret)]
                    if cnt > 0:
                        metric_dfs[m].loc[f"RD{rd}", f"RET{ret}"] = metric_sum[m][(rd, ret)] / cnt
                    else:
                        metric_dfs[m].loc[f"RD{rd}", f"RET{ret}"] = np.nan

        # 5. Define a helper for plotting + logging
        def _plot_and_log(df: pd.DataFrame, title: str):
            fig, ax = plt.subplots(figsize=(6, 5))
            cmap = plt.get_cmap("YlOrRd")
            mat = df.values.copy()
            im = ax.imshow(mat, interpolation="nearest", cmap=cmap, origin="upper")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(title)

            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_xticklabels(df.columns, fontsize=11)
            ax.set_yticks(np.arange(len(df.index)))
            ax.set_yticklabels(df.index, fontsize=11)

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if np.isnan(val):
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                                    facecolor="lightgray", edgecolor="none",
                                                    alpha=0.45, zorder=2))
                        ax.text(j, i, "-", ha='center', va='center', color='gray', fontsize=12, zorder=3)
                    else:
                        # format float with 2 decimals
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                                color='black', fontsize=12, zorder=3)

            for x in range(-1, len(df.columns)):
                ax.axvline(x + 0.5, color='white', linewidth=1)
            for y in range(-1, len(df.index)):
                ax.axhline(y + 0.5, color='white', linewidth=1)

            ax.set_title(title, fontsize=13, pad=12)
            ax.set_xlim(-0.5, len(df.columns)-0.5)
            ax.set_ylim(len(df.index)-0.5, -0.5)
            plt.tight_layout()

            # Log to W&B as image
            wandb.log({title.replace(" ", "_"): wandb.Image(fig)})
            plt.close(fig)

        # 6. Plot + log count heatmap
        _plot_and_log(count_df, f"{qtype}_num_samples_per_(ret,rd)")

        # 7. Plot + log each metric
        for m, df in metric_dfs.items():
            _plot_and_log(df, f"{qtype}_metric_{m}_per_(ret,rd)")