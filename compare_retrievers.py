
import pandas as pd
from bm25_baseline import BM25Retriever
from semantic_module import SemanticRetriever
from hybrid_retriever import HybridRetriever


from causal_metric import (
    causal_precision_at_k,
    causal_mrr,
    causal_ndcg
)

TOP_K = 5

def run_comparison(query):
    print(f"\n=== Running Retrieval Comparison for Query: '{query}' ===\n")
    
    
    bm25 = BM25Retriever()
    bm25_results = bm25.search(query, top_k=TOP_K)
    
    print("=== BM25 Top Results ===")
    for i, r in enumerate(bm25_results, 1):
        print(f"Rank {i}: PMID {r['pmid']}, Score: {r['bm25_score']:.4f}")
        print(f"Title: {r['title']}\n")
    
    
    semantic = SemanticRetriever()
    semantic_results = semantic.semantic_search(query, top_k=TOP_K)
    
    print("\n=== Semantic (BioBERT) Top Results ===")
    for i, r in enumerate(semantic_results, 1):
        print(f"Rank {i}: PMID {r['pmid']}, Score: {r['semantic_score']:.4f}")
        print(f"Title: {r['title']}\n")
    
    
    hybrid = HybridRetriever(alpha=0.6, beta=0.4, top_k=TOP_K)
    hybrid_results = hybrid.get_hybrid_results(query)
    
    print("\n=== Hybrid (Semantic + Causal) Top Results ===")
    for i, r in enumerate(hybrid_results, 1):
        print(f"Rank {i}: PMID {r['pmid']}, Hybrid Score: {r['hybrid_score']:.4f}")
        print(f"Title: {r['title']}")
        if r['supporting_sentences']:
            print("Supporting Causal Sentence(s):")
            for s in r['supporting_sentences']:
                print("-", s)
        print()
    
  
    cp10 = causal_precision_at_k(hybrid_results, k=TOP_K)
    cmrr = causal_mrr(hybrid_results)
    cndcg10 = causal_ndcg(hybrid_results, k=TOP_K)

    print("\n=== Causal Evaluation Metrics ===\n")
    print(f"Causal Precision@{TOP_K}: {cp10:.4f}")
    print(f"Causal MRR: {cmrr:.4f}")
    print(f"Causal NDCG@{TOP_K}: {cndcg10:.4f}")

    
    table_rows = []
    for i in range(TOP_K):
        row = {
            "Rank": i+1,
            "BM25 PMID": bm25_results[i]['pmid'] if i < len(bm25_results) else "",
            "BM25 Score": round(bm25_results[i]['bm25_score'], 4) if i < len(bm25_results) else "",
            "Semantic PMID": semantic_results[i]['pmid'] if i < len(semantic_results) else "",
            "Semantic Score": round(semantic_results[i]['semantic_score'], 4) if i < len(semantic_results) else "",
            "Hybrid PMID": hybrid_results[i]['pmid'] if i < len(hybrid_results) else "",
            "Hybrid Score": round(hybrid_results[i]['hybrid_score'], 4) if i < len(hybrid_results) else "",
            # ⭐ NEW: optional — add causal score for inspection
            "Causal Score": round(hybrid_results[i]['causal_score'], 4) if i < len(hybrid_results) else ""
        }
        table_rows.append(row)
    
    df = pd.DataFrame(table_rows)

    print(f"\n=== Side-by-Side Retrieval Comparison Table for Query: '{query}' ===\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    query = "How does adipose-tissue inflammation contribute to the development of insulin resistance?"
    run_comparison(query)
