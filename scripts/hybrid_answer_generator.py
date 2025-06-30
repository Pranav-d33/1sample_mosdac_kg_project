import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from difflib import get_close_matches
from prompt_builder import build_prompt, load_graph, find_node_id
from sentence_transformers import SentenceTransformer, util

# Load once globally
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # You can change this

def semantic_match_node(query, graph_nodes, model=embed_model, threshold=0.6):
    """
    Return the best matching node label and id from graph using semantic similarity.
    """
    node_labels = [node["label"] for node in graph_nodes if "label" in node]
    node_ids = [node["id"] for node in graph_nodes if "label" in node]

    query_embedding = model.encode(query, convert_to_tensor=True)
    label_embeddings = model.encode(node_labels, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, label_embeddings)[0]
    top_index = int(cosine_scores.argmax())

    if cosine_scores[top_index] >= threshold:
        return node_ids[top_index], node_labels[top_index]
    return None, None

# Load environment variables
load_dotenv()

# Configure Gemini API
if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# === CONFIG ===
INDEX_DIR = "outputs/vector_index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 6  # Increased to get more results including FAQs
MAX_HISTORY_TURNS = 5

def load_vector_resources():
    """Load FAISS index and associated data including content types"""
    index_path = f"{INDEX_DIR}/faiss.index"
    sources_path = f"{INDEX_DIR}/sources.json"
    corpus_path = f"{INDEX_DIR}/corpus.json"
    content_types_path = f"{INDEX_DIR}/content_types.json"
    
    # Check if files exist
    if not os.path.exists(index_path):
        print(f"❌ Vector index not found at {index_path}")
        print("Run 'python scripts/vector_retriever.py' first to build the index.")
        return None, [], [], []
    
    try:
        index = faiss.read_index(index_path)
        with open(sources_path, "r", encoding="utf-8") as f:
            sources = json.load(f)
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        
        # Load content types if available
        content_types = []
        if os.path.exists(content_types_path):
            with open(content_types_path, "r", encoding="utf-8") as f:
                content_types = json.load(f)
        else:
            content_types = ["document"] * len(corpus)  # Default to document type
        
        print(f"✅ Loaded vector index with {len(corpus)} items")
        doc_count = sum(1 for t in content_types if t == "document")
        faq_count = sum(1 for t in content_types if t == "faq")
        print(f"   📄 Documents: {doc_count}, ❓ FAQs: {faq_count}")
        
        return index, sources, corpus, content_types
    except Exception as e:
        print(f"❌ Error loading vector resources: {e}")
        return None, [], [], []

def get_top_chunks(query, index, corpus, sources, content_types, model):
    """Retrieve top K similar documents and FAQs using vector search"""
    if index is None or not corpus:
        return []
    
    try:
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
                    "score": float(D[0][len(results)])  # Distance score
                })
        
        # Separate and prioritize results
        faqs = [r for r in results if r["type"] in ("faq_complete", "faq_question_only")]
        docs = [r for r in results if r["type"] == "document"]
        
        print(f"📄 Retrieved {len(docs)} documents and ❓ {len(faqs)} FAQs")
        
        # Return mixed results (prioritize FAQs for direct questions)
        mixed_results = []
        if "?" in query or any(word in query.lower() for word in ["what", "how", "why", "when", "where"]):
            # Question detected - prioritize FAQs
            mixed_results.extend(faqs[:2])  # Top 2 FAQs
            mixed_results.extend(docs[:2])  # Top 2 docs
        else:
            # General query - prioritize documents
            mixed_results.extend(docs[:3])  # Top 3 docs
            mixed_results.extend(faqs[:1])  # Top 1 FAQ
        
        return mixed_results
    except Exception as e:
        print(f"❌ Error in vector search: {e}")
        return []

def extract_entities(query, graph_data):
    """Extract entities from query that match graph nodes"""
    if not graph_data or "nodes" not in graph_data:
        return []
    
    query_lower = query.lower()
    matched_entities = []
    
    # Check all nodes for matches
    for node in graph_data["nodes"]:
        node_id = node.get("id", "").lower()
        node_label = node.get("label", "").lower()
        
        # Check for exact or partial matches
        if (node_id in query_lower or node_label in query_lower or 
            any(word in node_id for word in query_lower.split()) or
            any(word in node_label for word in query_lower.split())):
            matched_entities.append(node.get("id"))
    
    # If no matches, try fuzzy matching on labels
    if not matched_entities:
        labels = [node.get("label", "") for node in graph_data["nodes"]]
        closest_labels = get_close_matches(query, labels, n=3, cutoff=0.6)
        for label in closest_labels:
            for node in graph_data["nodes"]:
                if node.get("label") == label:
                    matched_entities.append(node.get("id"))
    
    print(f"🔍 Found entities: {matched_entities}")
    return matched_entities[:3]

def get_triples(graph_data, entities):
    """Get relationship triples for given entities"""
    if not graph_data or not entities or "edges" not in graph_data:
        return []
    
    triples = []
    for entity in entities:
        # Find relationships where this entity is involved
        for edge in graph_data.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            relation = edge.get("relationship", "related_to")
            
            if source == entity:
                triples.append((source, relation, target))
            elif target == entity:
                triples.append((target, relation, source))
    
    print(f"📊 Found {len(triples)} graph relationships")
    return triples[:8]  # Limit to avoid overwhelming the prompt

def main():
    """Main chat loop"""
    print("🔧 Initializing Hybrid RAG + Graph + FAQ Assistant...")
    
    # Initialize components
    model = SentenceTransformer(MODEL_NAME)
    index, sources, corpus, content_types = load_vector_resources()
    graph_data = load_graph()
    
    if not graph_data.get("nodes"):
        print("⚠️ Warning: No graph data loaded. Only vector search will be available.")
    
    history = []
    print("\n🧠 Hybrid RAG + Graph + FAQ Assistant Ready!")
    print("Ask questions about Indian satellites, weather data, or general FAQs. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("🔍 Ask a question: ").strip()
            
            if query.lower() == "exit":
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\n🔄 Processing: {query}")
            
            # 1. Entity + Graph Search
            entities = extract_entities(query, graph_data)
            triples = get_triples(graph_data, entities) if entities else []
            
            # 2. Vector Search (including FAQs)
            top_chunks = get_top_chunks(query, index, corpus, sources, content_types, model)
            
            # 3. Build Prompt
            prompt = build_prompt(query, history, triples, top_chunks)
            
            # 4. Generate Response
            print("🤖 Generating response...")
            try:
                response = gemini_model.generate_content(prompt)
                answer = response.text.strip()
                
                print(f"\n📝 Answer:\n{answer}\n")
                
                # 5. Update History
                history.append((query, answer))
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()