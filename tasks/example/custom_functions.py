from typing import List, Dict, Optional, Any
from src.config_utils import RAGConfig, vprint


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _tokenize(text: str):
    return _normalize(text).split()


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
    vprint("[PREPROC] Starting dataset preprocessing...")
    processed = []
    if num_examples is not None:
        raw_examples = raw_examples[:num_examples]
    for i, ex in enumerate(raw_examples):
        # Example assumptions; change to what your dataset actually uses
        q = ex.get("question") or ex.get("query") or ex.get("input")
        gold = ex.get("answer") or ex.get("gold") or ex.get("target")

        if q is None or gold is None:
            vprint(f"[PREPROC][WARN] Skipping example {i} due to missing fields.")
            continue

        item = {
            "question": str(q),
            "gold": str(gold),
            "meta": {
                "idx": i,
                **{k: v for k, v in ex.items() if k not in {"question", "query", "input", "answer", "gold", "target"}},
            },
        }
        processed.append(item)

    vprint(f"[PREPROC] Completed preprocessing. Kept {len(processed)} examples.")
    return processed

def compute_metrics(
    prediction: str,
    gold: str,
    meta: Dict[str, Any],
    cfg: RAGConfig
) -> Dict[str, float]:
    """
    Custom metric function.
    Returns any number of metrics as a dict of {name: value}.

    This sample:
        - exact_match: 1.0 if normalized strings match, else 0.0
        - token_f1: simple F1 over token overlap
    """
    vprint("[METRICS] Computing metrics for one example...")

    pred_norm = _normalize(prediction)
    gold_norm = _normalize(gold)

    # Exact match (normalized)
    em = 1.0 if pred_norm == gold_norm else 0.0

    # Very simple token-level F1
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)

    if not pred_set and not gold_set:
        f1 = 1.0
    elif not pred_set or not gold_set:
        f1 = 0.0
    else:
        overlap = len(pred_set & gold_set)
        precision = overlap / len(pred_set)
        recall = overlap / len(gold_set)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return {
        "exact_match": em,
        "token_f1": f1,
    }