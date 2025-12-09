#!/usr/bin/env python3
"""
hybrid_retriever.py

Combines semantic and causal reasoning pipelines into a hybrid retrieval system,
with automatic fallback PubMed title + abstract retrieval for causal-only PMIDs.
"""

from semantic_module import SemanticRetriever
from causal_reasoner import CausalReasoner
from Bio import Entrez


def fetch_pubmed_title(pmid: str) -> str:
    """
    Fetch article title from PubMed for PMIDs missing from semantic index.
    Returns '[Title unavailable]' if retrieval fails.
    """
    Entrez.email = "your_email@example.com"  # TODO: replace with your email

    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=str(pmid),
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(handle)

        article_list = records.get("PubmedArticle", [])
        if not article_list:
            return "[Title unavailable]"

        article = article_list[0]["MedlineCitation"]["Article"]
        title = article.get("ArticleTitle", "")
        title = str(title).strip()

        return title if title else "[Title unavailable]"

    except Exception as e:
        print(f"[WARN] Failed to fetch title for PMID {pmid}: {e}")
        return "[Title unavailable]"



def fetch_pubmed_abstract(pmid: str) -> str:
    """
    Fetch article abstract from PubMed for PMIDs missing from semantic index.
    Returns '[Abstract unavailable]' if retrieval fails.
    """
    Entrez.email = "your_email@example.com"

    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=str(pmid),
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(handle)

        article_list = records.get("PubmedArticle", [])
        if not article_list:
            return "[Abstract unavailable]"

        article = article_list[0]["MedlineCitation"]["Article"]

        # abstract may contain multiple tagged sections
        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        if not abstract_parts:
            return "[Abstract unavailable]"

        text = " ".join(str(p).strip() for p in abstract_parts)
        return text if text else "[Abstract unavailable]"

    except Exception as e:
        print(f"[WARN] Failed to fetch abstract for PMID {pmid}: {e}")
        return "[Abstract unavailable]"




class HybridRetriever:
    """
    Hybrid retrieval system combining:
      - Semantic similarity scores from a BioBERT retriever
      - Causal evidence scores from a causal reasoning module
      - Title and abstract metadata fetched directly from PubMed when missing
    """

    def __init__(self, alpha=0.6, beta=0.4, top_k=10):
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k

        print("Initializing SemanticRetriever...")
        self.semantic_retriever = SemanticRetriever()

        print("Initializing CausalReasoner...")
        self.causal_reasoner = CausalReasoner()

        # Cache to avoid repeated network calls
        self.title_cache = {}
        self.abstract_cache = {}


   
    def get_title(self, pmid: str, sem_entry: dict) -> str:
        """Returns the best available title: semantic → cached → PubMed."""
        if "title" in sem_entry and sem_entry["title"] not in ("", None, "[Title unavailable]"):
            return sem_entry["title"]

        if pmid in self.title_cache:
            return self.title_cache[pmid]

        title = fetch_pubmed_title(pmid)
        self.title_cache[pmid] = title
        return title


    def get_abstract(self, pmid: str, sem_entry: dict) -> str:
        """Returns the best available abstract: semantic → cached → PubMed."""
        if "abstract" in sem_entry and sem_entry["abstract"] not in ("", None, "[Abstract unavailable]"):
            return sem_entry["abstract"]

        if pmid in self.abstract_cache:
            return self.abstract_cache[pmid]

        abstract = fetch_pubmed_abstract(pmid)
        self.abstract_cache[pmid] = abstract
        return abstract


   
    def get_hybrid_results(self, query):
        """
        Returns list of dicts:
            pmid, title, abstract,
            semantic_score, causal_score, hybrid_score,
            supporting_sentences
        """

        
        print("\nRunning semantic search...")
        semantic_results = self.semantic_retriever.semantic_search(query, top_k=self.top_k)
        pmid_to_semantic = {res["pmid"]: res for res in semantic_results}

        
        print("\nRunning causal reasoning...")
        causal_output = self.causal_reasoner.answer(query)

        causal_scores = {}
        pmid_to_sentences = {}

        for path_info in causal_output.get("top_explanations", []):
            path_score = path_info["score"]

            for rel in path_info["path"]["rels"]:
                pmid = rel["pmid"]
                sentence = rel["sentence"]

                if pmid not in causal_scores or path_score > causal_scores[pmid]:
                    causal_scores[pmid] = path_score
                    pmid_to_sentences[pmid] = [sentence]
                elif path_score == causal_scores[pmid]:
                    pmid_to_sentences[pmid].append(sentence)

       
        all_pmids = set(pmid_to_semantic.keys()) | set(causal_scores.keys())

       
        hybrid_results = []

        for pmid in all_pmids:
            sem = pmid_to_semantic.get(pmid, {})
            semantic_score = sem.get("semantic_score", 0.0)
            causal_score = causal_scores.get(pmid, 0.0)
            sentences = pmid_to_sentences.get(pmid, [])

            title = self.get_title(pmid, sem)
            abstract = self.get_abstract(pmid, sem)

            hybrid_score = self.alpha * semantic_score + self.beta * causal_score

            hybrid_results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "semantic_score": semantic_score,
                "causal_score": causal_score,
                "hybrid_score": hybrid_score,
                "supporting_sentences": sentences,
            })

       
       
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:self.top_k]



if __name__ == "__main__":
    print("Enter your biomedical query: ", end="")
    query = input()

    retriever = HybridRetriever(alpha=0.6, beta=0.4, top_k=10)
    results = retriever.get_hybrid_results(query)

    print("\n=== Hybrid Retrieval Results ===\n")
    for i, res in enumerate(results, 1):
        print(f"Rank {i}: PMID {res['pmid']}, Hybrid Score: {res['hybrid_score']:.4f}")
        print(f"Title: {res['title']}")
        print(f"Abstract: {res['abstract'][:300]}...")

        if res["supporting_sentences"]:
            print("Supporting Causal Sentence(s):")
            for s in res["supporting_sentences"]:
                print("-", s)

        print("-" * 80)
