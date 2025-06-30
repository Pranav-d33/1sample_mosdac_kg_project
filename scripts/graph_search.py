# scripts/graph_search.py

import networkx as nx
import json
from networkx.readwrite import json_graph

# Load Knowledge Graph
def load_graph(graph_path=r"outputs/phase2/knowledge_graph.json"):
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        G = json_graph.node_link_graph(data, edges="links")
    return G

# Get triples for matched entities
def get_triples(graph, entities, max_depth=1):
    triples = []

    for ent in entities:
        if ent not in graph:
            continue

        visited = set()
        queue = [(ent, 0)]
        
        while queue:
            current_node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            visited.add(current_node)

            for neighbor in graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                relation = graph[current_node][neighbor].get("relation", "related_to")
                triples.append((current_node, relation, neighbor))
                queue.append((neighbor, depth + 1))

    return triples

if __name__ == "__main__":
    graph = load_graph()

    # Step 1: Print first 20 nodes for inspection
    print("🔍 Sample nodes in the graph:")
    for i, node in enumerate(graph.nodes):
        print("-", node)
        if i > 20:
            break

    # Step 2: Fuzzy match input entity
    from difflib import get_close_matches
    query_entity = input("Enter entity to search: ")
    matched = get_close_matches(query_entity, list(graph.nodes), n=1, cutoff=0.6)

    if not matched:
        print("⚠️ No close match found for:", query_entity)
    else:
        entity = matched[0]
        print(f"✅ Closest match for '{query_entity}': {entity}")

        # Step 3: Get and display triples
        results = get_triples(graph, [entity])
        if not results:
            print("⚠️ No triples found.")
        else:
            print(f"\n📌 Triples for '{entity}':")
            for h, r, t in results[:10]:
                print(f"{h} --{r}--> {t}")
