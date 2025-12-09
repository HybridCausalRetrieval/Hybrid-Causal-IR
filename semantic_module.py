import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


class SemanticRetriever:
    def __init__(self,
                 faiss_index_file="./data/embeddings/semantic_index.faiss",
                 id_mapping_file="./data/embeddings/id_mapping.npy",
                 cleaned_csv="./data/cleaned/cleaned_abstracts.csv",
                 model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        # Load FAISS index
        print("Loading FAISS index...")
        self.index = faiss.read_index(faiss_index_file)
        
        # Load ID mapping
        self.pmid_mapping = np.load(id_mapping_file)
        
        # Load abstracts
        self.df = pd.read_csv(cleaned_csv).set_index('pmid')
        
        # Load embedding model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)


    def semantic_search(self, query, top_k=5):
        """Search FAISS for query and return top_k results with normalized scores"""
        # Embed query
        query_vec = self.model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vec)
        
        # Search FAISS
        distances, indices = self.index.search(query_vec, top_k)
        
        # Normalize scores to [0,1]
        scores = distances[0]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Collect results
        results = []
        for idx, score in zip(indices[0], scores):
            pmid = self.pmid_mapping[idx]
            row = self.df.loc[pmid]
            results.append({
                "pmid": pmid,
                "title": row['title'],
                "abstract": row['abstract'],
                "semantic_score": float(score)
            })
        return results
