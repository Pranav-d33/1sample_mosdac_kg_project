import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# Path configurations
CLEANED_DOCS_FILE = "outputs/cleaned_json/cleaned_docs.json"
FAQS_FILE = "outputs/cleaned_json/faqs.json"
INDEX_DIR = "outputs/vector_index"
os.makedirs(INDEX_DIR, exist_ok=True)

def load_chunks():
    chunks = []
    sources = []
    content_types = []

    # Load documents
    print(f"📄 Loading documents from: {CLEANED_DOCS_FILE}")
    if os.path.exists(CLEANED_DOCS_FILE):
        with open(CLEANED_DOCS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                print(f"   Documents file type: {type(data)}, length: {len(data) if isinstance(data, list) else 'N/A'}")
                
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", "") if isinstance(item, dict) else str(item)
                        if text and text.strip():
                            chunks.append(text)
                            sources.append(item.get("filename", "unknown") if isinstance(item, dict) else "unknown")
                            content_types.append("document")
                else:
                    print(f"   ⚠️ Unexpected document format: {type(data)}")
            except Exception as e:
                print(f"⚠️ Error loading {CLEANED_DOCS_FILE}: {e}")
    else:
        print(f"   ❌ Documents file not found")

    # Load FAQs - include questions even without answers
    print(f"❓ Loading FAQs from: {FAQS_FILE}")
    if os.path.exists(FAQS_FILE):
        file_size = os.path.getsize(FAQS_FILE)
        print(f"   FAQ file size: {file_size} bytes")
        
        if file_size > 0:
            with open(FAQS_FILE, "r", encoding="utf-8") as f:
                try:
                    faqs = json.load(f)
                    print(f"   FAQ file type: {type(faqs)}, length: {len(faqs) if isinstance(faqs, (list, dict)) else 'N/A'}")
                    
                    if isinstance(faqs, list):
                        for i, faq in enumerate(faqs):
                            if isinstance(faq, dict):
                                question = (faq.get("question") or faq.get("q") or 
                                          faq.get("Question") or faq.get("Q") or "").strip()
                                answer = (faq.get("answer") or faq.get("a") or 
                                        faq.get("Answer") or faq.get("A") or "").strip()
                                
                                # Include questions even if answers are empty
                                if question:
                                    if answer:
                                        # Full Q&A pair
                                        combined_text = f"Question: {question}\nAnswer: {answer}"
                                        faq_type = "faq_complete"
                                    else:
                                        # Question only - mark for LLM to note missing answer
                                        combined_text = f"Frequently Asked Question: {question}\n[Note: This is a common user question that needs to be answered based on available knowledge]"
                                        faq_type = "faq_question_only"
                                    
                                    chunks.append(combined_text)
                                    sources.append(f"FAQ: {question[:50]}...")
                                    content_types.append(faq_type)
                                    print(f"   ✅ Added FAQ: {question[:50]}... ({'with answer' if answer else 'question only'})")
                                else:
                                    print(f"   ⚠️ FAQ {i+1} has empty question")
                    
                    elif isinstance(faqs, dict):
                        # Handle dictionary format
                        for key, value in faqs.items():
                            if key.strip():
                                if isinstance(value, str) and value.strip():
                                    combined_text = f"Question: {key}\nAnswer: {value}"
                                    faq_type = "faq_complete"
                                else:
                                    combined_text = f"Frequently Asked Question: {key}\n[Note: This is a common user question that needs to be answered based on available knowledge]"
                                    faq_type = "faq_question_only"
                                
                                chunks.append(combined_text)
                                sources.append(f"FAQ: {key[:50]}...")
                                content_types.append(faq_type)
                                print(f"   ✅ Added dict FAQ: {key[:50]}... ({'with answer' if isinstance(value, str) and value.strip() else 'question only'})")
                    
                    else:
                        print(f"   ⚠️ Unexpected FAQ format: {type(faqs)}")
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON decode error: {e}")
                except Exception as e:
                    print(f"   ❌ Error processing FAQs: {e}")
        else:
            print(f"   ❌ FAQ file is empty")
    else:
        print(f"   ❌ FAQ file not found")

    doc_count = len([c for c, t in zip(chunks, content_types) if t == "document"])
    faq_complete_count = len([c for c, t in zip(chunks, content_types) if t == "faq_complete"])
    faq_question_count = len([c for c, t in zip(chunks, content_types) if t == "faq_question_only"])
    
    print(f"\n📊 Summary:")
    print(f"📄 Loaded {doc_count} documents")
    print(f"❓ Loaded {faq_complete_count} complete FAQs")
    print(f"❔ Loaded {faq_question_count} FAQ questions (without answers)")
    print(f"📊 Total chunks: {len(chunks)}")
    
    return chunks, sources, content_types

def build_faiss_index(chunks, sources, content_types, model_name="BAAI/bge-small-en-v1.5"):
    if not chunks:
        print("❌ No chunks to index!")
        return
    
    print(f"🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"🔢 Encoding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks, batch_size=16, show_progress_bar=True, convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    
    # Save metadata
    with open(os.path.join(INDEX_DIR, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)
    
    with open(os.path.join(INDEX_DIR, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    
    with open(os.path.join(INDEX_DIR, "content_types.json"), "w", encoding="utf-8") as f:
        json.dump(content_types, f, indent=2)

    print(f"✅ FAISS index built and saved to {INDEX_DIR}")
    doc_count = len([t for t in content_types if t == "document"])
    faq_complete_count = len([t for t in content_types if t == "faq_complete"])
    faq_question_count = len([t for t in content_types if t == "faq_question_only"])
    print(f"   📄 Documents: {doc_count}")
    print(f"   ❓ Complete FAQs: {faq_complete_count}")
    print(f"   ❔ FAQ Questions: {faq_question_count}")

if __name__ == "__main__":
    chunks, sources, content_types = load_chunks()
    if chunks:
        build_faiss_index(chunks, sources, content_types)
    else:
        print("❌ No content loaded! Check your input files.")