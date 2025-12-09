# causal_metrics.py

import math

def has_causal_evidence(result):
    """Return True if the retrieved document contains causal evidence."""
    return len(result.get("supporting_sentences", [])) > 0


def causal_precision_at_k(results, k=10):
    if len(results) == 0:
        return 0.0
    top_k = results[:k]
    causal_hits = sum(1 for r in top_k if has_causal_evidence(r))
    return causal_hits / k

def causal_mrr(results):
    for i, r in enumerate(results, start=1):
        if has_causal_evidence(r):
            return 1.0 / i
    return 0.0  # no causal docs found


def causal_ndcg(results, k=10):
    if len(results) == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, r in enumerate(results[:k], start=1):
        rel = r.get("causal_score", 0.0)
        dcg += rel / math.log2(i + 1)

    # IDCG (ideal DCG sorting by causal_score)
    ideal = sorted(results[:k], key=lambda x: x.get("causal_score", 0.0), reverse=True)
    idcg = 0.0
    for i, r in enumerate(ideal, start=1):
        rel = r.get("causal_score", 0.0)
        idcg += rel / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0
