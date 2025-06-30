# scripts/query_retriever.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "outputs/vector_index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5

def load_index():
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/sources.json", "r", encoding="utf-8") as f:
        sources = json.load(f)
    return index, sources

def search_query(query, index, sources, model):
    query_vector = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vector, k=TOP_K)

    results = []
    for i in I[0]:
        if i < len(sources):
            results.append({
                "source": sources[i],
                "text": corpus[i]  # pulled from the embedded text corpus
            })
    return results

if __name__ == "__main__":
    import os

    # Load corpus
    CORPUS_FILE = os.path.join(INDEX_DIR, "corpus.json")
    if not os.path.exists(CORPUS_FILE):
        print("❌ Missing corpus.json — please create it during vector indexing.")
        exit()

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    model = SentenceTransformer(MODEL_NAME)
    index, sources = load_index()

    print("🧠 Semantic Search Engine Ready")
    while True:
        query = input("\n🔍 Ask a question (or 'exit'): ")
        if query.strip().lower() == "exit":
            break

        results = search_query(query, index, sources, model)
        print("\n📄 Top Results:")
        for i, res in enumerate(results):
            print(f"\n#{i+1} — Source: {res['source']}\n{res['text'][:500]}...\n---")
