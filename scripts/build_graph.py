import json
import os
import networkx as nx

INPUT_FILE = "outputs/phase2/entities.json"
GRAPHML_OUTPUT = "outputs/phase2/knowledge_graph.graphml"
JSON_OUTPUT = "outputs/phase2/knowledge_graph.json"

import re

def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")  # remove non-utf8
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)  # remove control chars
    return text.strip()

def build_knowledge_graph(entities_data):
    G = nx.Graph()

    for doc in entities_data:
        filename = sanitize_text(doc["filename"])
        entities = doc.get("entities", {})

        flat_entities = []
        for label, values in entities.items():
            for val in values:
                cleaned_val = sanitize_text(val)
                if cleaned_val:
                    flat_entities.append((cleaned_val, label))

        for value, label in flat_entities:
            G.add_node(value, label=sanitize_text(label))

        for i in range(len(flat_entities)):
            for j in range(i + 1, len(flat_entities)):
                a, a_label = flat_entities[i]
                b, b_label = flat_entities[j]
                if a != b:
                    G.add_edge(a, b, source=filename)

    return G

def save_graph(graph):
    os.makedirs(os.path.dirname(GRAPHML_OUTPUT), exist_ok=True)
    print("💾 Saving graph...")

    try:
        nx.write_graphml(graph, GRAPHML_OUTPUT)
        print(f"✅ GraphML saved → {GRAPHML_OUTPUT}")
    except Exception as e:
        print(f"⚠️ Skipped GraphML due to error: {e}")

    data = nx.readwrite.json_graph.node_link_data(graph)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Graph JSON saved → {JSON_OUTPUT}")


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        entities_data = json.load(f)

    graph = build_knowledge_graph(entities_data)
    print(f"📊 Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    save_graph(graph)

if __name__ == "__main__":
    main()
