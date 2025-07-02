import spacy
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

INPUT_FILES = [
    "outputs/cleaned_json/cleaned_docs.json",
    "outputs/cleaned_json/site_data.json"
]
OUTPUT_FILE = "outputs/phase2/entities.json"

# Load spaCy model and dynamically add patterns
def load_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Generate patterns dynamically from site_data.json
    patterns = generate_patterns_from_data("outputs/cleaned_json/site_data.json")
    ruler.add_patterns(patterns)
    
    return nlp

def generate_patterns_from_data(file_path):
    """Generate entity patterns dynamically from site_data.json."""
    patterns = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                title = item.get("title", "").strip()
                mission_details = item.get("mission_details", "").strip()
                if title:
                    patterns.append({"label": "TITLE", "pattern": title})
                if mission_details:
                    patterns.append({"label": "MISSION_DETAILS", "pattern": mission_details})
    return patterns

nlp = load_spacy_model()

def sanitize_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())  # Normalize whitespace
    return text

def extract_entities(text):
    """Extract entities using spaCy."""
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text.strip())
    return entities

def process_document(item):
    """Process a single document."""
    text = sanitize_text(item.get("text", ""))
    if text:
        entities = extract_entities(text)
        return {
            "source": item.get("filename", "unknown"),
            "source_type": "document",
            "entities": dict(entities)
        }
    return None

def process_site_data(item):
    """Process a single site data item."""
    combined_text = ""
    item_type = item.get("type", "unknown")
    
    # Combine all relevant fields for entity extraction
    combined_text += sanitize_text(item.get("title", "")) + "\n"
    combined_text += sanitize_text(item.get("mission_details", "")) + "\n"
    for meta in item.get("meta", []):
        combined_text += sanitize_text(meta.get("content", "")) + "\n"
    for table in item.get("tables", []):
        if isinstance(table, dict):
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            combined_text += sanitize_text(" | ".join(headers)) + "\n"
            for row in rows:
                combined_text += sanitize_text(" | ".join(row)) + "\n"
        elif isinstance(table, list):
            for row in table:
                combined_text += sanitize_text(" | ".join(row)) + "\n"
    for aria in item.get("aria_labels", []):
        combined_text += sanitize_text(aria.get("aria-label", "") + " " + aria.get("text", "")) + "\n"
    for catalog in item.get("product_catalog", []):
        combined_text += sanitize_text(" | ".join(catalog)) + "\n"
    for paragraph in item.get("paragraphs", []):
        combined_text += sanitize_text(paragraph) + "\n"
    for image_link in item.get("image_links", []):
        combined_text += sanitize_text(image_link) + "\n"
    
    if combined_text.strip():
        entities = extract_entities(combined_text)
        return {
            "source": item.get("url", "unknown"),
            "source_type": item_type,
            "entities": dict(entities),
            "raw_data": combined_text.strip()  # Include raw combined text for reference
        }
    return None

def run_entity_extraction():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output = []

    def process_file(input_file):
        results = []
        if not os.path.exists(input_file):
            print(f"⚠️ File not found: {input_file}")
            return results
            
        with open(input_file, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            if "cleaned_docs" in input_file:
                result = process_document(item)
            elif "site_data" in input_file:
                result = process_site_data(item)
            else:
                result = None
            
            if result:
                results.append(result)
        return results

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, input_file) for input_file in INPUT_FILES]
        for future in futures:
            output.extend(future.result())

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    doc_count = len([x for x in output if x["source_type"] == "document"])
    dataset_count = len([x for x in output if x["source_type"] == "dataset"])
    site_count = len([x for x in output if x["source_type"] == "site_page"])
    
    print(f"✅ Extracted entities from {len(output)} sources:")
    print(f"   📄 Documents: {doc_count}")
    print(f"   📊 Dataset pages: {dataset_count}")
    print(f"   🏗️ Site pages: {site_count}")
    print(f"   💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_entity_extraction()