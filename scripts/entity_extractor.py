import json
import spacy
from difflib import get_close_matches

def load_graph():
    """Load the knowledge graph"""
    try:
        with open("outputs/knowledge_graph_normalized.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return {"nodes": [], "edges": []}

def extract_entities(query, graph_nodes):
    """Extract entities from query that match graph nodes"""
    if not graph_nodes:
        return []
    
    query_lower = query.lower()
    matched_entities = []
    
    # Check each graph node for matches
    for node in graph_nodes:
        if isinstance(node, dict):
            node_id = node.get("id", "").lower()
            node_label = node.get("label", "").lower()
        else:
            node_id = str(node).lower()
            node_label = str(node).lower()
        
        # Check for matches
        if (node_id in query_lower or node_label in query_lower or 
            any(word in node_id for word in query_lower.split()) or
            any(word in node_label for word in query_lower.split())):
            matched_entities.append(node_id if isinstance(node, dict) else node)
    
    # If no direct matches, try fuzzy matching
    if not matched_entities:
        node_labels = [node.get("label", node) if isinstance(node, dict) else str(node) for node in graph_nodes]
        closest = get_close_matches(query, node_labels, n=3, cutoff=0.6)
        matched_entities.extend(closest)
    
    return list(set(matched_entities))  # Remove duplicates

if __name__ == "__main__":
    graph_data = load_graph()
    graph_nodes = graph_data.get("nodes", [])
    
    query = input("Enter your query: ")
    entities = extract_entities(query, graph_nodes)
    print(f"Found entities: {entities}")