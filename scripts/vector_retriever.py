import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# Path configurations
CLEANED_DOCS_FILE = "outputs/cleaned_json/cleaned_docs.json"
FAQS_FILE = "outputs/cleaned_json/faqs.json"
SITE_DATA_FILE = "outputs/cleaned_json/site_data.json"
INDEX_DIR = "outputs/vector_index"
os.makedirs(INDEX_DIR, exist_ok=True)

def load_all_data():
    """
    Load and consolidate text chunks from:
    - cleaned_docs.json  (documents)
    - faqs.json          (FAQs)
    - site_data.json     (site crawl)
    Returns three parallel lists: texts, sources, and types.
    """
    chunks = []
    sources = []
    content_types = []

    # 1. Documents
    if os.path.exists(CLEANED_DOCS_FILE):
        with open(CLEANED_DOCS_FILE, "r", encoding="utf-8") as f:
            docs = json.load(f)
            for item in docs:
                text = item.get("text", "").strip()
                if text:
                    chunks.append(text)
                    sources.append(item.get("filename", "document"))
                    content_types.append("document")

    # 2. FAQs
    if os.path.exists(FAQS_FILE):
        with open(FAQS_FILE, "r", encoding="utf-8") as f:
            faqs = json.load(f)
            for faq in faqs:
                q = faq.get("question", "").strip()
                a = faq.get("answer", "").strip()
                if q:
                    if a:
                        text = f"Question: {q}\nAnswer: {a}"
                        faq_type = "faq_complete"
                    else:
                        text = f"Frequently Asked Question: {q}\n[Answer missing]"
                        faq_type = "faq_question_only"
                    chunks.append(text)
                    sources.append(f"FAQ: {q[:50]}...")
                    content_types.append(faq_type)

    # 3. Site Data
    if os.path.exists(SITE_DATA_FILE):
        with open(SITE_DATA_FILE, "r", encoding="utf-8") as f:
            pages = json.load(f)
            for page in pages:
                # Skip pages with errors
                if "error" in page:
                    continue
                    
                url = page.get("url", "")
                page_type = page.get("type", "unknown")
                parts = []

                if page_type == "dataset":
                    # Handle dataset page format
                    title = page.get("title", "").strip()
                    if title:
                        parts.append(f"Title: {title}")
                    
                    # Add paragraphs
                    for paragraph in page.get("paragraphs", []):
                        if paragraph.strip():
                            parts.append(paragraph.strip())
                    
                    # Add tables (dataset format: list of lists)
                    for table in page.get("tables", []):
                        if isinstance(table, list):
                            for row in table:
                                if isinstance(row, list):
                                    parts.append(" | ".join(str(cell) for cell in row))
                    
                    # Add raw data
                    raw_data = page.get("raw_data", "").strip()
                    if raw_data:
                        parts.append(f"Raw Data: {raw_data}")
                
                elif page_type == "site_page":
                    # Handle site page format
                    # Mission details
                    md = page.get("mission_details", "").strip()
                    if md:
                        parts.append(md)

                    # Meta tags
                    for m in page.get("meta", []):
                        key = m.get("key", "").strip()
                        val = m.get("content", "").strip()
                        if key and val:
                            parts.append(f"{key}: {val}")

                    # Tables (site page format: dict with headers and rows)
                    for table in page.get("tables", []):
                        if isinstance(table, dict):
                            headers = table.get("headers", [])
                            if headers:
                                parts.append(" | ".join(headers))
                            for row in table.get("rows", []):
                                if isinstance(row, list):
                                    parts.append(" | ".join(str(cell) for cell in row))

                    # Product catalog
                    for prod in page.get("product_catalog", []):
                        if isinstance(prod, list):
                            parts.append(" | ".join(str(cell) for cell in prod))

                    # ARIA labels
                    for aria in page.get("aria_labels", []):
                        aria_label = aria.get("aria-label", "").strip()
                        aria_text = aria.get("text", "").strip()
                        if aria_label or aria_text:
                            parts.append(f"{aria_label} {aria_text}".strip())
                    
                    # Add raw data
                    raw_data = page.get("raw_data", "").strip()
                    if raw_data:
                        parts.append(f"Raw Data: {raw_data}")

                combined = "\n".join(parts).strip()
                if combined:
                    chunks.append(combined)
                    sources.append(f"{page_type}: {url}")
                    content_types.append(page_type)

    return chunks, sources, content_types


def build_faiss_index(chunks, sources, content_types, model_name="BAAI/bge-small-en-v1.5"):
    if not chunks:
        print("❌ No chunks to index! Check your input files.")
        return

    print(f"🤖 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"🔢 Encoding {len(chunks)} text chunks...")
    embeddings = model.encode(
        chunks, batch_size=16, show_progress_bar=True, convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    # Save sources, corpus, and content types
    with open(os.path.join(INDEX_DIR, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)
    with open(os.path.join(INDEX_DIR, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    with open(os.path.join(INDEX_DIR, "content_types.json"), "w", encoding="utf-8") as f:
        json.dump(content_types, f, indent=2)

    print(f"✅ FAISS index built and saved to `{INDEX_DIR}`")
    print(f"   📄 Documents: {sum(1 for t in content_types if t=='document')}")
    print(f"   ❓ FAQ complete: {sum(1 for t in content_types if t=='faq_complete')}")
    print(f"   ❔ FAQ questions: {sum(1 for t in content_types if t=='faq_question_only')}")
    print(f"   📊 Dataset pages: {sum(1 for t in content_types if t=='dataset')}")
    print(f"   🏗️ Site pages: {sum(1 for t in content_types if t=='site_page')}")


if __name__ == "__main__":
    chunks, sources, content_types = load_all_data()
    build_faiss_index(chunks, sources, content_types)