import json
import networkx as nx
from pathlib import Path

GRAPH_PATH = "outputs/knowledge_graph_normalized.json"

def load_graph(path=GRAPH_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)

def normalize_graph(graph):
    normalized_nodes = []
    normalized_edges = []

    for node_id, attrs in graph.nodes(data=True):
        # Use all metadata directly from node attributes
        node_data = {"id": node_id}
        node_data.update(attrs)
        normalized_nodes.append(node_data)

    for source, target, attrs in graph.edges(data=True):
        edge_data = {
            "source": source,
            "target": target,
            "label": attrs.get("label", "related_to")
        }
        normalized_edges.append(edge_data)

    return {"nodes": normalized_nodes, "edges": normalized_edges}

def save_normalized_graph(normalized_graph, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_graph, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    graph = load_graph()
    normalized_graph = normalize_graph(graph)
    save_normalized_graph(normalized_graph, "outputs/knowledge_graph_normalized.json")