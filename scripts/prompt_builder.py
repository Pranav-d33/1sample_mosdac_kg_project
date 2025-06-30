import json
import difflib
import networkx as nx
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer, util

# Load once (can be optimized further)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# Paths and constants
GRAPH_PATH = Path("outputs/knowledge_graph_normalized.json")
MAX_HISTORY_TURNS = 5
MAX_TRIPLES = 8
MAX_CHUNKS = 3

def load_graph():
    """Load the knowledge graph"""
    try:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return {"nodes": [], "edges": []}

def find_node_id(query, graph_data):
    import difflib

    nodes = graph_data["nodes"]
    labels = [node["label"] for node in nodes]
    label_to_id = {node["label"]: node["id"] for node in nodes}

    # Step 1: Fuzzy match
    closest = difflib.get_close_matches(query, labels, n=1, cutoff=0.6)
    if closest:
        label = closest[0]
        return label_to_id[label], label

    # Step 2: Semantic match fallback
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    label_embs = embed_model.encode(labels, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, label_embs)[0]
    top_idx = int(cos_scores.argmax())
    if cos_scores[top_idx] > 0.6:
        label = labels[top_idx]
        return label_to_id[label], label

    return None, None


def extract_node_facts(graph_data, node_id):
    """Extract attributes and relationships for a specific node"""
    attributes = []
    triples = []
    
    # Find the node
    node = None
    for n in graph_data["nodes"]:
        if n["id"] == node_id:
            node = n
            break
    
    if node:
        # Extract node attributes
        for key, value in node.items():
            if key not in ["id", "label"]:
                if isinstance(value, list):
                    attributes.append(f"• {key}: {', '.join(map(str, value))}")
                else:
                    attributes.append(f"• {key}: {value}")
    
    # Extract relationships (edges)
    for edge in graph_data.get("edges", []):
        if edge.get("source") == node_id:
            triples.append((edge["source"], edge.get("relationship", "related_to"), edge["target"]))
        elif edge.get("target") == node_id:
            triples.append((edge["target"], edge.get("relationship", "related_to"), edge["source"]))
    
    return attributes, triples

def filter_graph_facts(triples, node_id):
    """Format triples into readable facts"""
    facts = []
    for subject, relation, obj in triples[:MAX_TRIPLES]:
        if subject == node_id:
            facts.append(f"• {subject} --{relation}--> {obj}")
        else:
            facts.append(f"• {obj} <--{relation}-- {subject}")
    return facts

def format_mixed_content(top_chunks):
    """Format mixed content (documents + FAQs) for better presentation"""
    if not top_chunks:
        return "No relevant information found."
    
    formatted_sections = []
    
    # Separate different content types
    complete_faqs = [chunk for chunk in top_chunks if chunk.get("type") == "faq_complete"]
    question_only_faqs = [chunk for chunk in top_chunks if chunk.get("type") == "faq_question_only"]
    docs = [chunk for chunk in top_chunks if chunk.get("type") == "document"]
    
    # Format complete FAQs first (highest priority)
    if complete_faqs:
        formatted_sections.append("❓ **Relevant FAQs with Answers:**")
        for i, faq in enumerate(complete_faqs):
            text = faq.get("text", "")
            formatted_sections.append(f"\n**FAQ {i+1}:**\n{text}")
    
    # Format question-only FAQs (these indicate common questions users ask)
    if question_only_faqs:
        formatted_sections.append("\n❔ **Common User Questions (answer based on available knowledge):**")
        for i, faq in enumerate(question_only_faqs):
            text = faq.get("text", "")
            # Extract just the question part
            question_part = text.split('\n[Note:')[0].replace('Frequently Asked Question: ', '')
            formatted_sections.append(f"\n**User often asks:** {question_part}")
    
    # Format Documents section
    if docs:
        formatted_sections.append("\n📄 **Technical Documentation:**")
        for i, doc in enumerate(docs):
            text = doc.get("text", "")[:1200]  # Truncate long documents
            source = doc.get("source", "unknown")
            formatted_sections.append(f"\n**[Document {i+1} - {source}]**\n{text}")
    
    return "\n".join(formatted_sections)

def build_prompt(query, history, triples, top_chunks):
    """Build the complete prompt for the LLM with enhanced FAQ support"""
    
    # Format history
    history_str = ""
    if history:
        history_items = []
        for q, a in history[-MAX_HISTORY_TURNS:]:
            history_items.append(f"User: {q}\nAssistant: {a}")
        history_str = "\n\n".join(history_items)
    
    # Format triples
    triple_str = ""
    if triples:
        triple_items = []
        for subject, relation, obj in triples[:MAX_TRIPLES]:
            triple_items.append(f"• {subject} --{relation}--> {obj}")
        triple_str = "\n".join(triple_items)
    
    # Format mixed content (docs + FAQs)
    context_str = format_mixed_content(top_chunks)
    
    # Try to find entity information
    graph_data = load_graph()
    node_id, label = find_node_id(query, graph_data)
    
    entity_info = ""
    if node_id:
        attributes, node_triples = extract_node_facts(graph_data, node_id)
        if attributes:
            entity_info = f"\n🛰️ {label} Information:\n" + "\n".join(attributes)
        # Add node triples to the general triples
        triples.extend(node_triples)
        triple_str = "\n".join([f"• {s} --{r}--> {o}" for s, r, o in triples[:MAX_TRIPLES]])
    
    # Check for question-only FAQs in results
    question_only_faqs = [chunk for chunk in top_chunks if chunk.get("type") == "faq_question_only"]
    faq_guidance = ""
    if question_only_faqs:
        faq_guidance = "\n💡 **Note:** Some frequently asked questions are listed above. Please provide comprehensive answers to these common queries using the available technical documentation and graph facts."
    
    # Build the final prompt
    prompt = f"""You are a helpful assistant for India's satellite data portal (MOSDAC).
Your job is to provide detailed, factually accurate, and structured answers using the context below.

Guidelines:
- Answer based on the available technical documentation and graph facts
- If FAQs with answers are provided, use them as verified information
- If common user questions (without answers) are listed, provide comprehensive answers using available knowledge
- For satellite/product questions, describe: monitoring capabilities, applications, instruments, data products
- Be specific and technical when appropriate
- If information is incomplete, acknowledge this and provide what you can

{entity_info}

📌 Graph Facts:
{triple_str if triple_str else 'No specific graph relationships found.'}

📚 **Knowledge Base:**
{context_str}

{faq_guidance}

🕰️ Previous Conversation:
{history_str if history_str else 'This is the start of our conversation.'}

📨 Current Question:
{query}

Please provide a comprehensive answer based on the available information."""

    return prompt