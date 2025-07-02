import json
import networkx as nx
from pathlib import Path
import re

GRAPH_PATH = "outputs/phase2/knowledge_graph.json"
OUTPUT_PATH = "outputs/knowledge_graph_normalized.json"

def sanitize_text(text):
    """Sanitize text to remove invalid characters and normalize whitespace."""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # Remove control characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def validate_node_attributes(attrs):
    """Validate and sanitize node attributes."""
    sanitized_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, str):
            sanitized_attrs[key] = sanitize_text(value)
        elif isinstance(value, list):
            sanitized_attrs[key] = [sanitize_text(v) for v in value if isinstance(v, str)]
        else:
            sanitized_attrs[key] = value
    return sanitized_attrs

def validate_edge_attributes(attrs):
    """Validate and sanitize edge attributes."""
    sanitized_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, str):
            sanitized_attrs[key] = sanitize_text(value)
        elif isinstance(value, list):
            sanitized_attrs[key] = [sanitize_text(v) for v in value if isinstance(v, str)]
        else:
            sanitized_attrs[key] = value
    return sanitized_attrs

def load_graph(path=GRAPH_PATH):
    """Load the graph from JSON format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")

def normalize_graph(graph):
    """Normalize the graph structure."""
    normalized_nodes = []
    normalized_edges = []
    seen_nodes = set()
    seen_edges = set()

    for node_id, attrs in graph.nodes(data=True):
        # Sanitize and validate node attributes
        sanitized_attrs = validate_node_attributes(attrs)
        node_data = {"id": sanitize_text(node_id)}
        node_data.update(sanitized_attrs)

        # Avoid duplicate nodes
        if node_data["id"] not in seen_nodes:
            normalized_nodes.append(node_data)
            seen_nodes.add(node_data["id"])

    for source, target, attrs in graph.edges(data=True):
        # Sanitize and validate edge attributes
        sanitized_attrs = validate_edge_attributes(attrs)
        edge_data = {
            "source": sanitize_text(source),
            "target": sanitize_text(target),
            "relationship": sanitized_attrs.get("relationship", "related_to"),
            "weight": sanitized_attrs.get("weight", 1)  # Default weight
        }
        edge_data.update({k: v for k, v in sanitized_attrs.items() if k not in ["relationship", "weight"]})

        # Avoid duplicate edges
        edge_key = (edge_data["source"], edge_data["target"], edge_data["relationship"])
        if edge_key not in seen_edges:
            normalized_edges.append(edge_data)
            seen_edges.add(edge_key)

    return {"nodes": normalized_nodes, "edges": normalized_edges}

def save_normalized_graph(normalized_graph, output_path):
    """Save the normalized graph to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_graph, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print("🔄 Loading knowledge graph...")
    graph = load_graph()

    print("🔧 Normalizing graph structure...")
    normalized_graph = normalize_graph(graph)

    save_normalized_graph(normalized_graph, OUTPUT_PATH)

    print(f"✅ Normalized graph saved to: {OUTPUT_PATH}")
    print(f"📊 Nodes: {len(normalized_graph['nodes'])}")
    print(f"🔗 Edges: {len(normalized_graph['edges'])}")