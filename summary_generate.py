#!/usr/bin/env python3
"""
hybrid_retriever_with_summary.py

Hybrid retrieval (semantic + causal) with OpenAI-generated relevance summaries.
"""

"""
Hybrid retrieval (semantic + causal) with Gemini 2.5 Flash relevance summaries.
"""

#!/usr/bin/env python3
"""
Hybrid retrieval (semantic + causal) with OpenAI-generated relevance summaries.

Requirements:
- pip install openai pandas python-dotenv fuzzywuzzy scispacy spacy sentence-transformers neo4j
- Add your OpenAI API key in .env:
    OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX
"""


from hybrid_retriever import HybridRetriever
import os
from dotenv import load_dotenv
import openai

# ---------------------------
# Load OpenAI API key
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ---------------------------
# Combined summary generator using OpenAI
# ---------------------------
def generate_combined_summary(query: str, docs: list) -> str:
    """
    Generate a single summary for multiple documents explaining why they are relevant to the query,
    highlighting causal relationships.
    """
    # Combine titles and abstracts into one text block
    combined_text = ""
    for i, doc in enumerate(docs, 1):
        combined_text += f"Document {i}:\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\n\n"

    prompt = f"""
You are a biomedical AI assistant.

Query: "{query}"

Here are the retrieved documents:
{combined_text}

Task: In 3-5 sentences, summarize why these documents are relevant to the query,
highlighting causal relationships mentioned across the documents.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-4o", "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a helpful biomedical assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )
        summary = response.choices[0].message['content'].strip()
    except Exception as e:
        print("Error generating summary:", e)
        summary = "[Summary unavailable]"
    
    return summary

# ---------------------------
# Main pipeline
# ---------------------------
if __name__ == "__main__":
    query = "How does adipose-tissue inflammation contribute to the development of insulin resistance?"
    retriever = HybridRetriever(alpha=0.6, beta=0.4, top_k=5)

    # Step 1: Retrieve documents
    results = retriever.get_hybrid_results(query)

    # Step 2: Store documents
    stored_docs = []
    print("\n=== Hybrid Retrieval Results ===\n")
    for i, res in enumerate(results, 1):
        doc = {
            "pmid": res['pmid'],
            "title": res.get("title", "[Title unavailable]"),
            "abstract": res.get("abstract", "[Abstract unavailable]")
        }
        stored_docs.append(doc)

        print(f"Rank {i}: PMID {doc['pmid']}, Hybrid Score: {res['hybrid_score']:.4f}")
        print(f"Title: {doc['title']}")
        print(f"Abstract: {doc['abstract'][:300]}...")
        print("-"*80)

    # Step 3: Generate a single combined summary
    final_summary = generate_combined_summary(query, stored_docs)
    print("\n=== Combined Relevance Summary ===\n")
    print(final_summary)

