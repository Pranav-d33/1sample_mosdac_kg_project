import os
import json
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import faiss
from difflib import get_close_matches
from scripts.prompt_builder import build_prompt, load_graph

# Load environment variables
load_dotenv()

# Configure Gemini API
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Constants
INDEX_DIR = "outputs/vector_index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 6

@st.cache_resource(show_spinner=False)
def load_vector_resources():
    index_path = f"{INDEX_DIR}/faiss.index"
    sources_path = f"{INDEX_DIR}/sources.json"
    corpus_path = f"{INDEX_DIR}/corpus.json"
    content_types_path = f"{INDEX_DIR}/content_types.json"

    if not os.path.exists(index_path):
        st.error(f"Vector index not found at {index_path}. Please build the index first.")
        return None, [], [], []

    index = faiss.read_index(index_path)
    with open(sources_path, "r", encoding="utf-8") as f:
        sources = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    if os.path.exists(content_types_path):
        with open(content_types_path, "r", encoding="utf-8") as f:
            content_types = json.load(f)
    else:
        content_types = ["document"] * len(corpus)

    return index, sources, corpus, content_types

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_sentence_transformer_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_graph_data():
    return load_graph()

def semantic_match_node(query, graph_nodes, model, threshold=0.6):
    node_labels = [node["label"] for node in graph_nodes if "label" in node]
    node_ids = [node["id"] for node in graph_nodes if "label" in node]

    query_embedding = model.encode(query, convert_to_tensor=True)
    label_embeddings = model.encode(node_labels, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, label_embeddings)[0]
    top_index = int(cosine_scores.argmax())

    if cosine_scores[top_index] >= threshold:
        return node_ids[top_index], node_labels[top_index]
    return None, None

def get_top_chunks(query, index, corpus, sources, content_types, model):
    if index is None or not corpus:
        return []

    query_vector = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vector, k=TOP_K)

    results = []
    for i in I[0]:
        if i < len(corpus):
            content_type = content_types[i] if i < len(content_types) else "document"
            results.append({
                "source": sources[i] if i < len(sources) else "unknown",
                "text": corpus[i],
                "type": content_type,
                "score": float(D[0][len(results)])
            })

    faqs = [r for r in results if r["type"] in ("faq_complete", "faq_question_only")]
    docs = [r for r in results if r["type"] == "document"]
    site_data = [r for r in results if r["type"] == "site_data"]
    raw_data = [r for r in results if r["type"] == "raw_data"]

    mixed_results = []
    if "?" in query or any(word in query.lower() for word in ["what", "how", "why", "when", "where"]):
        mixed_results.extend(faqs[:2])
        mixed_results.extend(site_data[:2])
        mixed_results.extend(raw_data[:2])
        mixed_results.extend(docs[:2])
    else:
        mixed_results.extend(docs[:3])
        mixed_results.extend(site_data[:2])
        mixed_results.extend(raw_data[:1])
        mixed_results.extend(faqs[:1])

    return mixed_results

def extract_entities(query, graph_data):
    if not graph_data or "nodes" not in graph_data:
        return []

    query_lower = query.lower()
    matched_entities = []

    for node in graph_data["nodes"]:
        node_id = node.get("id", "").lower()
        node_label = node.get("label", "").lower()

        if (node_id in query_lower or node_label in query_lower or
            any(word in node_id for word in query_lower.split()) or
            any(word in node_label for word in query_lower.split())):
            matched_entities.append(node.get("id"))

    if not matched_entities:
        labels = [node.get("label", "") for node in graph_data["nodes"]]
        closest_labels = get_close_matches(query, labels, n=3, cutoff=0.6)
        for label in closest_labels:
            for node in graph_data["nodes"]:
                if node.get("label") == label:
                    matched_entities.append(node.get("id"))

    return matched_entities[:3]

def get_triples(graph_data, entities):
    if not graph_data or not entities or "edges" not in graph_data:
        return []

    triples = []
    for entity in entities:
        for edge in graph_data.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            relation = edge.get("relationship", "related_to")

            if source == entity:
                triples.append((source, relation, target))
            elif target == entity:
                triples.append((target, relation, source))

    return triples[:8]

def generate_response(query, history, index, corpus, sources, content_types, model, graph_data):
    entities = extract_entities(query, graph_data)
    triples = get_triples(graph_data, entities) if entities else []
    top_chunks = get_top_chunks(query, index, corpus, sources, content_types, model)
    prompt = build_prompt(query, history, triples, top_chunks)

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating response: {e}"

    return answer

def main():
    st.title("Hybrid RAG + Graph + FAQ Assistant")

    if "history" not in st.session_state:
        st.session_state.history = []

    index, sources, corpus, content_types = load_vector_resources()
    embed_model = load_embedding_model()
    model = load_sentence_transformer_model()
    graph_data = load_graph_data()

    query = st.text_input("Ask a question about Indian satellites, weather data, or general FAQs:")

    if query:
        with st.spinner("Generating response..."):
            answer = generate_response(query, st.session_state.history, index, corpus, sources, content_types, model, graph_data)
            st.session_state.history.append((query, answer))

    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

if __name__ == "__main__":
    main()