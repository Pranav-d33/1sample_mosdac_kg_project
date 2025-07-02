import json
import os
import re
from collections import defaultdict
import networkx as nx

INPUT_FILE = "outputs/phase2/entities.json"
OUTPUT_DIR = "outputs/phase2"
GRAPH_JSON = os.path.join(OUTPUT_DIR, "knowledge_graph.json")
GRAPH_GRAPHML = os.path.join(OUTPUT_DIR, "knowledge_graph.graphml")

def sanitize_for_xml(text):
    """Sanitize text to be XML-compatible."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Remove control characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')  # Ensure valid UTF-8
    return text.strip()

def build_knowledge_graph():
    """Build a knowledge graph from extracted entities."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ Entities file not found: {INPUT_FILE}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load entities
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        entities_data = json.load(f)
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Track entity co-occurrences
    co_occurrences = defaultdict(int)
    entity_sources = defaultdict(set)
    entity_types = defaultdict(set)
    entity_raw_data = defaultdict(set)
    
    print("🔨 Building knowledge graph...")
    
    for item in entities_data:
        source = item.get("source", "unknown")
        source_type = item.get("source_type", "unknown")
        entities = item.get("entities", {})
        raw_data = item.get("raw_data", "")
        
        # Flatten all entities from this source
        all_entities = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # Sanitize entity name
                entity = sanitize_for_xml(entity.strip())
                if len(entity) > 2:  # Filter out very short entities
                    all_entities.append(entity)
                    entity_sources[entity].add(source)
                    entity_types[entity].add(entity_type)
                    entity_raw_data[entity].add(raw_data)
        
        # Create co-occurrence relationships
        for i, entity1 in enumerate(all_entities):
            for entity2 in all_entities[i+1:]:
                if entity1 != entity2:
                    pair = tuple(sorted([entity1, entity2]))
                    co_occurrences[pair] += 1
    
    # Add nodes to graph (convert lists to strings for GraphML compatibility)
    for entity in entity_sources.keys():
        # Sanitize all attributes for XML compatibility
        sanitized_sources = [sanitize_for_xml(s) for s in entity_sources[entity]]
        sanitized_types = [sanitize_for_xml(t) for t in entity_types[entity]]
        sanitized_raw_data = [sanitize_for_xml(d) for d in entity_raw_data[entity]]
        
        G.add_node(sanitize_for_xml(entity), 
                  sources="; ".join(sanitized_sources),  # Convert list to string
                  types="; ".join(sanitized_types),      # Convert list to string
                  raw_data="; ".join(sanitized_raw_data),  # Include raw data
                  source_count=len(entity_sources[entity]))
    
    # Add edges based on co-occurrence (with threshold)
    min_cooccurrence = 2  # Minimum co-occurrence to create edge
    
    for (entity1, entity2), count in co_occurrences.items():
        if count >= min_cooccurrence:
            G.add_edge(sanitize_for_xml(entity1), sanitize_for_xml(entity2), 
                      weight=count,
                      relationship="co_occurs_with")
    
    # Create output in JSON format (keep lists here for better JSON structure)
    graph_data = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "sources_processed": len(entities_data),
            "min_cooccurrence_threshold": min_cooccurrence
        }
    }
    
    # Add nodes (restore lists for JSON output)
    for node in G.nodes():
        graph_data["nodes"].append({
            "id": node,
            "label": node,
            "sources": list(entity_sources.get(node, [])),     # Keep as list for JSON
            "types": list(entity_types.get(node, [])),         # Keep as list for JSON
            "raw_data": list(entity_raw_data.get(node, [])),   # Include raw data
            "source_count": len(entity_sources.get(node, []))
        })
    
    # Add edges
    for source, target, attrs in G.edges(data=True):
        graph_data["edges"].append({
            "source": source,
            "target": target,
            "weight": attrs.get("weight", 1),
            "relationship": attrs.get("relationship", "related_to")
        })
    
    # Save JSON format
    with open(GRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    # Save GraphML format (for tools like Gephi, Cytoscape)
    try:
        nx.write_graphml(G, GRAPH_GRAPHML)
        print(f"📁 GraphML saved to: {GRAPH_GRAPHML}")
    except Exception as e:
        print(f"⚠️ Could not save GraphML: {e}")
        print("📁 JSON format saved successfully instead")
    
    # Print statistics
    print(f"✅ Knowledge graph built successfully!")
    print(f"📊 Nodes: {G.number_of_nodes()}")
    print(f"🔗 Edges: {G.number_of_edges()}")
    print(f"📁 JSON saved to: {GRAPH_JSON}")
    
    # Print top entities by source count
    top_entities = sorted(entity_sources.items(), 
                         key=lambda x: len(x[1]), reverse=True)[:10]
    
    print(f"\n🔝 Top entities by source count:")
    for entity, sources in top_entities:
        print(f"   {entity}: {len(sources)} sources")
    
    return graph_data

if __name__ == "__main__":
    build_knowledge_graph()