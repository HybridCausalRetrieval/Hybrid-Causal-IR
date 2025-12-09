#!/usr/bin/env python3
"""
causal_reasoner.py

Advanced causal reasoning module for Neo4j-based biomedical causal graphs.
(Option C – spaCy NER + SciSpaCy + SBERT embeddings + fuzzy matching)

This file implements:

1. Query → Concept Mapping
   - spaCy for noun chunking
   - SciSpaCy biomedical NER
   - fuzzy matching
   - SBERT semantic similarity fallback

2. Graph Search
   - shortest causal path
   - multi-path causal exploration
   - upstream/downstream search

3. Path Scoring
   - path length
   - evidence count from CAUSES edges
   - path redundancy
   - trigger strength

4. High-Level API
   - answer causal “WHY?” queries
   - return causal paths with explanation
   - compute causal relevance for semantic retrieval

"""

import os
import re
import numpy as np
from typing import List, Dict, Any

from neo4j import GraphDatabase
from dotenv import load_dotenv

import spacy
import scispacy
import fuzzywuzzy
from fuzzywuzzy import process

from sentence_transformers import SentenceTransformer, util as sbert_util

#
import pandas as pd



load_dotenv()

#NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password1234")

# spaCy model for noun-phrase extraction
print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

# SciSpaCy biomedical NER (optional but powerful)
try:
    print("Loading SciSpaCy...")
    biomed_nlp = spacy.load("en_ner_bionlp13cg_md")
except:
    print("SciSpaCy model not found. Install via: pip install scispacy && download model")
    biomed_nlp = nlp


print("Loading SBERT...")
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))




def load_all_concepts():
    """Load all concept IDs & names from Neo4j."""
    query = "MATCH (c:Concept) RETURN c.id AS id, c.name AS name"
    with get_driver().session() as session:
        records = session.run(query)
        return [(r["id"], r["name"]) for r in records]


ALL_CONCEPTS = load_all_concepts()
ALL_CONCEPT_NAMES = [name for _, name in ALL_CONCEPTS]
ALL_CONCEPT_IDS = [cid for cid, _ in ALL_CONCEPTS]



def extract_query_terms(query: str) -> List[str]:
    """Extract noun phrases + biomedical entities from the query."""
    doc = nlp(query)
    biom = biomed_nlp(query)

    terms = set()

    # Basic noun chunks
    for chunk in doc.noun_chunks:
        terms.add(chunk.text.lower())

    # Biomed entities
    for ent in biom.ents:
        terms.add(ent.text.lower())

    # Clean terms
    clean_terms = []
    for t in terms:
        t = re.sub(r"[^a-zA-Z0-9\- ]", "", t).strip()
        if 2 <= len(t) <= 50:
            clean_terms.append(t)

    return clean_terms


def fuzzy_map(term: str, threshold=80):
    """Fuzzy match a query term to concept names."""
    match, score = process.extractOne(term, ALL_CONCEPT_NAMES)
    if score >= threshold:
        idx = ALL_CONCEPT_NAMES.index(match)
        return ALL_CONCEPT_IDS[idx], match, score
    return None



def embed_map(term: str, top_k=3):
    """Semantic similarity using SBERT embeddings."""
    if not ALL_CONCEPT_NAMES:
        return []

    # Get embeddings
    term_emb = sbert.encode(term, convert_to_tensor=True)
    concept_embs = sbert.encode(ALL_CONCEPT_NAMES, convert_to_tensor=True)

    # Compute cosine similarity
    scores = sbert_util.cos_sim(term_emb, concept_embs)[0]

    # Check if scores is empty
    if scores.numel() == 0:
        return []

    # Get top indices safely
    sorted_indices = np.argsort(scores.cpu().numpy())[::-1]
    if len(sorted_indices) == 0:
        return []

    top_results = sorted_indices[:top_k]

    mapped = []
    for idx in top_results:
        mapped.append({
            "concept_id": ALL_CONCEPT_IDS[idx],
            "name": ALL_CONCEPT_NAMES[idx],
            "score": float(scores[idx])
        })
    return mapped

def map_query_to_concepts(query: str) -> List[str]:
    """Map query terms to concept IDs using:
       - fuzzy matching
       - semantic embedding fallback"""
    terms = extract_query_terms(query)
    mapped_concepts = []

    for t in terms:
        # Try fuzzy match
        fm = fuzzy_map(t)
        if fm:
            mapped_concepts.append(fm[0])
            continue

        # Otherwise fall back to SBERT semantic match
        em = embed_map(t)
        if em and em[0]["score"] > 0.35:
            mapped_concepts.append(em[0]["concept_id"])

    return list(set(mapped_concepts))



def find_paths(source_id: str, target_id: str, max_depth=4):
    """Find paths between concepts."""
    query = f"""
    MATCH p = (a:Concept {{id: $sid}})-[:CAUSES*1..{max_depth}]->(b:Concept {{id: $tid}})
    RETURN nodes(p) AS nodes, relationships(p) AS rels
    LIMIT 20
    """
    with get_driver().session() as session:
        results = session.run(query, sid=source_id, tid=target_id)
        paths = []
        for r in results:
            nodes = [{"id": n["id"], "name": n["name"]} for n in r["nodes"]]
            rels = [{"pmid": rel["pmid"], "trigger": rel["trigger"], "sentence": rel["sentence"]} 
                    for rel in r["rels"]]
            paths.append({"nodes": nodes, "rels": rels})
        return paths



def trigger_strength(trigger: str) -> float:
    """Map causal trigger verbs to strengths."""
    strong = ["induces", "causes", "activates"]
    moderate = ["increases", "upregulates", "promotes"]
    weak = ["associated", "linked", "related"]

    trigger = trigger.lower()
    if trigger in strong:
        return 1.0
    if trigger in moderate:
        return 0.6
    if trigger in weak:
        return 0.3
    return 0.4  # default


def score_path(path):
    """Compute weighted causal score."""
    length = len(path["rels"])
    
    # 1) Distance score (shorter is better)
    distance_score = 1 / (1 + length)

    # 2) Evidence score (number of PMIDs)
    evidence_score = len(path["rels"]) / 5  # normalized

    # 3) Path redundancy (not known here, assume 1)
    path_count_score = 1.0

    # 4) Trigger strength average
    ts = np.mean([trigger_strength(r["trigger"]) for r in path["rels"]])

    # Final weighted score
    final_score = (
        0.4 * distance_score +
        0.3 * evidence_score +
        0.2 * path_count_score +
        0.1 * ts
    )
    return float(final_score)

def format_explanations(result, top_k=10):
    """Neatly format the causal reasoning output for readability."""
    query = result["query"]
    concepts = result["mapped_concepts"]
    exps = result["top_explanations"][:top_k]

    output = []
    output.append("=== Causal Reasoning Result ===\n")
    output.append(f"Query: {query}\n")
    output.append("Mapped Concepts:")
    for c in concepts:
        output.append(f"  - {c}")
    output.append("\nTop Explanations:\n")

    rank = 1
    for exp in exps:
        src = exp["source"]
        tgt = exp["target"]
        score = round(exp["score"], 3)

        # Clean target names (remove noisy concepts)
        if len(tgt) < 3 or tgt.startswith(("p-", ">", "non-", "both")):
            continue

        rel = exp["path"]["rels"][0]
        trigger = rel["trigger"]
        pmid = rel["pmid"]
        sentence = rel["sentence"]

        output.append(f"{rank}. {src} → {tgt}")
        output.append(f"   Trigger: {trigger}")
        output.append(f"   Evidence PMID: {pmid}")
        #output.append(f"   Title: {exp['title']}")
        #output.append(f"   Abstract: {exp['abstract'][:300]}...")
        output.append(f"   Evidence sentence: {sentence}")
        output.append(f"   Score: {score}\n")
        rank += 1

    return "\n".join(output)





class CausalReasoner:

    def __init__(self):
        print("CausalReasoner initialized.")
    
    def answer(self, query: str):
        """Main entry point for “WHY” questions."""
        print("Mapping query to concepts...")
        query_concepts = map_query_to_concepts(query)

        if not query_concepts:
            return {"error": "No relevant concepts found in query."}

        print(f"Mapped concepts: {query_concepts}")

        # For each concept, get downstream explanations
        results = []

        for qc in query_concepts:
            # Explore all downstream connections up to depth 3
            for tid in ALL_CONCEPT_IDS:
                paths = find_paths(qc, tid, max_depth=3)
                if not paths:
                    continue

                for p in paths:
                    score = score_path(p)
                    
                    results.append({
                        "source": qc,
                        "target": tid,
                        "path": p,
                        "score": score
                    })

        # Sort by descending causal strength
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "mapped_concepts": query_concepts,
            "top_explanations": results[:10]
        }



