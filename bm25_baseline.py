# bm25_baseline_simple.py
import pandas as pd
from rank_bm25 import BM25Okapi
import re

def simple_tokenize(text):
    """Lowercase + split on non-alphanumeric characters"""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

class BM25Retriever:
    def __init__(self, cleaned_csv="./data/cleaned/cleaned_abstracts.csv"):
        self.df = pd.read_csv(cleaned_csv).set_index('pmid')
        self.tokenized_corpus = [simple_tokenize(abstract) for abstract in self.df['abstract']]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=5):
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]
        scores = scores / scores.max()

        results = []
        for idx in top_indices:
            pmid = self.df.index[idx]
            results.append({
                "pmid": pmid,
                "title": self.df.loc[pmid, "title"],
                "abstract": self.df.loc[pmid, "abstract"],
                "bm25_score": float(scores[idx])
            })
        return results


