import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer, util
from prompt_builder import build_prompt, load_graph  # assumes you updated prompt_builder to handle site_data
from difflib import get_close_matches

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

INDEX_DIR    = "outputs/vector_index"
MODEL_EMBED  = "BAAI/bge-small-en-v1.5"     # single consistent embedding model
TOP_K        = 6
MAX_HISTORY  = 5
SIM_THRESHOLD = 0.6                     # for semantic graph matching

# ── Initialize embedding model once ─────────────────────────────────────────
embed_model = SentenceTransformer(MODEL_EMBED)

# ── Utility: semantic graph node match ───────────────────────────────────────
def semantic_match_node(query, graph_nodes, threshold=SIM_THRESHOLD):
    labels = [n["label"] for n in graph_nodes]
    ids    = [n["id"]    for n in graph_nodes]
    q_embed = embed_model.encode(query, convert_to_tensor=True)
    lbl_emb = embed_model.encode(labels, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(q_embed, lbl_emb)[0]
    idx  = int(sims.argmax().item())
    if sims[idx] >= threshold:
        return ids[idx], labels[idx]
    return None, None

# ── Load vector resources ────────────────────────────────────────────────────
def load_vector_resources():
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    src_path = os.path.join(INDEX_DIR, "sources.json")
    corp_path= os.path.join(INDEX_DIR, "corpus.json")
    type_path= os.path.join(INDEX_DIR, "content_types.json")

    if not os.path.exists(idx_path):
        raise RuntimeError("❌ Vector index missing. Run vector_retriever first.")

    index   = faiss.read_index(idx_path)
    sources = json.load(open(src_path, "r", encoding="utf-8"))
    corpus  = json.load(open(corp_path,"r", encoding="utf-8"))
    types   = json.load(open(type_path,"r", encoding="utf-8"))
    print(f"✅ Loaded {len(corpus)} chunks ({sources.count('document')} docs, {sum(t.startswith('faq') for t in types)} faqs, {types.count('site_data')} site pages)")
    return index, sources, corpus, types

# ── Retrieve top chunks ──────────────────────────────────────────────────────
def get_top_chunks(query, index, corpus, sources, types):
    # embed & search
    q_vec = embed_model.encode([query], convert_to_numpy=True)
    D, I  = index.search(q_vec, k=TOP_K)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx >= len(corpus): continue
        results.append({
            "source": sources[idx],
            "text":   corpus[idx],
            "type":   types[idx],
            "score":  float(D[0][rank])
        })

    # separate by type
    docs      = [r for r in results if r["type"] == "document"]
    faqs      = [r for r in results if r["type"].startswith("faq")]
    site_data = [r for r in results if r["type"] == "site_data"]
    raw_data  = [r for r in results if r["type"] == "raw_data"]

    # heuristics: if it's a question, prefer faqs then site_data then raw_data then docs
    if "?" in query.lower() or any(w in query.lower() for w in ["what", "how", "why", "when", "where"]):
        return (faqs[:2] + site_data[:2] + raw_data[:2] + docs[:2])
    # else prefer docs then site_data then raw_data
    return (docs[:3] + site_data[:2] + raw_data[:1] + faqs[:1])

# ── Extract entities from query ──────────────────────────────────────────────
def extract_entities(query, graph_data):
    # First try semantic match
    node_id, _ = semantic_match_node(query, graph_data.get("nodes", []))
    if node_id:
        return [node_id]
    # fallback fuzzy substring match
    ql = query.lower()
    found = [n["id"] for n in graph_data.get("nodes", [])
             if n.get("id", "").lower() in ql or n.get("label", "").lower() in ql]
    if not found:
        labels = [n["label"] for n in graph_data.get("nodes", [])]
        close = get_close_matches(query, labels, n=1, cutoff=0.6)
        if close:
            found = [n["id"] for n in graph_data["nodes"] if n["label"] == close[0]]
    print(f"🔍 Entities matched: {found}")
    return found[:3]

# ── Get graph triples ───────────────────────────────────────────────────────
def get_triples(graph_data, entities):
    triples = []
    edges = graph_data.get("edges", [])
    for e in entities:
        for edge in edges:
            if edge["source"] == e:
                triples.append((e, edge.get("relationship", "related_to"), edge["target"]))
            elif edge["target"] == e:
                triples.append((e, edge.get("relationship", "related_to"), edge["source"]))
    print(f"📊 Extracted {len(triples)} triples")
    return triples[:8]

# ── Main interactive loop ───────────────────────────────────────────────────
def main():
    print("🔧 Initializing RAG+Graph Assistant…")
    # load models & data
    index, sources, corpus, types = load_vector_resources()
    graph_data = load_graph()

    history = []
    print("🧠 Ready! Type 'exit' to quit.")
    while True:
        query = input("\n🔍 You: ").strip()
        if not query: continue
        if query.lower() == "exit":
            print("👋 Bye!"); break

        # 1) Graph entities & triples
        ents   = extract_entities(query, graph_data)
        triples= get_triples(graph_data, ents) if ents else []

        # 2) Vector RAG
        chunks = get_top_chunks(query, index, corpus, sources, types)

        # 3) Build LLM prompt
        prompt = build_prompt(query, history, triples, chunks)

        # 4) Generate
        print("🤖 Thinking…")
        try:
            resp = gemini_model.generate_content(prompt)
            answer = resp.text.strip()
            print(f"\n📝 Assistant:\n{answer}")
            history.append((query, answer))
            if len(history) > MAX_HISTORY:
                history.pop(0)
        except Exception as e:
            print("❌ LLM error:", e)

if __name__ == "__main__":
    main()