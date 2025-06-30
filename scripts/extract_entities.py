import spacy
import json
import os
from collections import defaultdict

INPUT_FILE = "outputs/cleaned_json/cleaned_docs.json"
OUTPUT_FILE = "outputs/phase2/entities.json"

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text.strip())
    return entities

def run_entity_extraction():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    output = []
    for item in docs:
        text = item["text"]
        entities = extract_entities(text)
        output.append({
            "filename": item["filename"],
            "entities": dict(entities)
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted entities from {len(output)} documents → {OUTPUT_FILE}")

if __name__ == "__main__":
    run_entity_extraction()
