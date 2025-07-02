import json
import spacy
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer, util
import re
from collections import defaultdict
import functools

# ── Configuration ────────────────────────────────────────────────────────────
GRAPH_FILE        = "outputs/knowledge_graph_normalized.json"
EMBED_MODEL       = "BAAI/bge-small-en-v1.5"
SIM_THRESHOLD     = 0.4  
FALLBACK_THRESHOLD= 0.25  
TOP_K_SEMANTIC    = 10    
FUZZY_CUTOFF      = 0.3   

# ── Load models once ────────────────────────────────────────────────────────
embed_model = SentenceTransformer(EMBED_MODEL)
nlp = spacy.load("en_core_web_sm")

# ── Load the graph JSON ──────────────────────────────────────────────────────
def load_graph():
    """Load the normalized knowledge graph."""
    try:
        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return {"nodes": [], "edges": []}

# ── Dynamic text normalization and variation generation ──────────────────────
def generate_text_variations(text):
    """Automatically generate text variations using multiple strategies."""
    if not text:
        return []
    
    variations = set()
    text = str(text).strip()
    variations.add(text)
    variations.add(text.lower())
    
    # 1. Punctuation and separator variations
    no_punct = re.sub(r'[^\w\s]', ' ', text)
    variations.add(no_punct)
    variations.add(no_punct.lower())
    
    # Replace different separators
    for sep in ['-', '_', '/', '.', ':']:
        variations.add(text.replace(sep, ' '))
        variations.add(text.replace(sep, ''))
        variations.add(text.replace(' ', sep))
    
    # 2. Acronym generation
    words = re.findall(r'\b\w+\b', text)
    if len(words) > 1:
        # First letter acronym
        acronym = ''.join(word[0].upper() for word in words if word)
        variations.add(acronym)
        variations.add(acronym.lower())
        
        # First letter with dashes/spaces
        variations.add('-'.join(word[0].upper() for word in words if word))
        variations.add(' '.join(word[0].upper() for word in words if word))
    
    # 3. Number format variations
    # Convert spelled numbers to digits and vice versa
    number_map = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
    }
    
    text_lower = text.lower()
    for num_word, num_digit in number_map.items():
        if num_word in text_lower:
            variations.add(text_lower.replace(num_word, num_digit))
    
    # 4. Whitespace normalization
    normalized = re.sub(r'\s+', ' ', text).strip()
    variations.add(normalized)
    variations.add(normalized.lower())
    
    # 5. Remove common prefixes/suffixes
    prefixes = ['the ', 'a ', 'an ', 'mission ', 'satellite ', 'project ']
    suffixes = [' mission', ' satellite', ' project', ' program']
    
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            variations.add(text[len(prefix):])
    
    for suffix in suffixes:
        if text_lower.endswith(suffix):
            variations.add(text[:-len(suffix)])
    
    # Remove empty and very short variations
    variations = {v.strip() for v in variations if v.strip() and len(v.strip()) > 1}
    
    return list(variations)

# ── Build searchable index from graph ───────────────────────────────────────
@functools.lru_cache(maxsize=1)
def build_search_index(graph_nodes):
    """Build a comprehensive search index with all variations."""
    search_index = defaultdict(set)  # variation -> set of node_ids
    
    for node in graph_nodes:
        node_id = node.get("id", "") if isinstance(node, dict) else str(node)
        node_label = node.get("label", "") if isinstance(node, dict) else str(node)
        
        # Generate variations for both id and label
        all_variations = set()
        all_variations.update(generate_text_variations(node_id))
        all_variations.update(generate_text_variations(node_label))
        
        # Add to search index
        for variation in all_variations:
            search_index[variation.lower()].add(node_id)
    
    return search_index

# ── Enhanced matching using the search index ────────────────────────────────
def match_with_index(query, search_index):
    """Match query against pre-built search index."""
    query_variations = generate_text_variations(query)
    matched_ids = set()
    
    for variation in query_variations:
        variation_lower = variation.lower()
        
        # Exact match in index
        if variation_lower in search_index:
            matched_ids.update(search_index[variation_lower])
        
        # Partial matches
        for indexed_var in search_index:
            if variation_lower in indexed_var or indexed_var in variation_lower:
                matched_ids.update(search_index[indexed_var])
    
    return list(matched_ids)

# ── Semantic matching with embeddings ───────────────────────────────────────
def semantic_match_nodes(query, graph_nodes, threshold=SIM_THRESHOLD):
    """Enhanced semantic matching."""
    if not graph_nodes:
        return []
    
    # Prepare texts for embedding
    texts = []
    node_ids = []
    
    for node in graph_nodes:
        node_id = node.get("id", "") if isinstance(node, dict) else str(node)
        node_label = node.get("label", "") if isinstance(node, dict) else str(node)
        
        # Use both original and some key variations for embedding
        base_texts = [node_label, node_id]
        variations = generate_text_variations(node_label)[:3]  # Limit to top 3
        base_texts.extend(variations)
        
        for text in base_texts:
            if text.strip():
                texts.append(text.strip())
                node_ids.append(node_id)
    
    if not texts:
        return []
    
    # Generate query variations for semantic matching
    query_variations = generate_text_variations(query)[:5]  # Limit to top 5
    
    all_matches = set()
    
    for q_var in query_variations:
        try:
            q_emb = embed_model.encode(q_var, convert_to_tensor=True)
            text_emb = embed_model.encode(texts, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(q_emb, text_emb)[0]
            
            # Get top matches
            top_scores, top_idxs = sims.topk(min(TOP_K_SEMANTIC, len(texts)))
            
            for score, idx in zip(top_scores, top_idxs):
                if score.item() >= threshold:
                    all_matches.add(node_ids[idx.item()])
        
        except Exception as e:
            print(f"Warning: Semantic matching failed for '{q_var}': {e}")
            continue
    
    return list(all_matches)

# ── Main extraction function ─────────────────────────────────────────────────
def extract_entities(query, graph_nodes):
    """Extract entities using multiple strategies."""
    if not graph_nodes:
        return []

    matched = set()
    
    # Strategy 1: Pre-built search index
    search_index = build_search_index(tuple(graph_nodes))  # Cache index
    index_matches = match_with_index(query, search_index)
    matched.update(index_matches)
    
    # Strategy 2: Semantic matching
    if len(matched) < 3:
        semantic_matches = semantic_match_nodes(query, graph_nodes)
        matched.update(semantic_matches)
    
    return list(matched)