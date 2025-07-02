# 🛰️ MOSDAC Knowledge Graph & Hybrid RAG System
A sophisticated pipeline that builds a Knowledge Graph from the MOSDAC portal, powering a hybrid Retrieval-Augmented Generation (RAG) system for intelligent Q&A.

---

## ✨ Features

-   **🌐 Automated Content Crawling**: Fetches documents and FAQs directly from the MOSDAC portal.
-   **🧠 Advanced Entity Extraction**: Uses `spaCy` and custom rules to identify and extract key entities and relationships.
-   **🕸️ Dynamic Knowledge Graph**: Constructs a comprehensive graph from extracted data, representing complex connections.
-   **🔍 Hybrid Semantic Search**: Combines traditional keyword search with FAISS-powered vector search for superior retrieval accuracy.
-   **🤖 AI-Powered Answers**: Leverages Google's Gemini model to generate context-aware, human-like answers based on retrieved information.
-   **🖥️ Interactive UI**: A user-friendly Streamlit application for easy interaction and querying.

---

## 🛠️ Technology Stack

-   **Backend**: Python
-   **AI/ML**: Google Generative AI (Gemini), SentenceTransformers, spaCy, FAISS
-   **Data Handling**: Pandas, NetworkX
-   **Web Scraping**: BeautifulSoup, Selenium
-   **Application**: Streamlit

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

-   Python 3.9 or higher
-   Google API Key

### 2. Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/your-repo/mosdac-kg-project.git
cd mosdac-kg-project

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables
# Create a .env file in the root directory and add your key:
# GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

### 3. Running the Pipeline

Execute the scripts in the following order to crawl data, build the knowledge graph, and start the application.

| Step | Command                               | Description                               |
| :--- | :------------------------------------ | :---------------------------------------- |
| 1.   | `python scripts/crawl_documents.py`   | Crawls document links from MOSDAC.        |
| 2.   | `python scripts/crawl_faqs.py`        | Crawls FAQs from the MOSDAC website.      |
| 3.   | `python scripts/download_documents.py`| Downloads the crawled documents.          |
| 4.   | `python scripts/extract_text_from_docs.py` | Extracts text from PDF documents.      |
| 5.   | `python scripts/extract_entities.py`  | Extracts entities from all text sources.  |
| 6.   | `python scripts/build_graph.py`       | Builds the knowledge graph.               |
| 7.   | `python scripts/normalize_graph.py`   | Normalizes the graph for consistency.     |
| 8.   | `python scripts/vector_retriever.py`  | Creates the FAISS index for semantic search. |
| 9.   | `streamlit run streamlit_app.py`      | **Starts the interactive web application.** |

---

## 📂 Project Structure

The repository is organized as follows:

```
mosdac_kg_project/
├── configs/              # Configuration files (e.g., portal_config.json)
├── data/                 # Raw data, including downloaded documents
├── outputs/              # All generated outputs (JSON, graphs, indexes)
├── scripts/              # All Python scripts for the pipeline
├── .env                  # Environment variables (needs to be created)
├── requirements.txt      # Project dependencies
├── streamlit_app.py      # The main Streamlit application
└── README.md             # This file
```

---

**Note:** Some utility scripts (like `prompt_builder.py`, `entity_extractor.py`) are typically imported as modules and not run directly.

---

## Future Work

- **Frontend Integration:** A frontend interface will be added in future commits.
- **API/Service Layer:** Exposing backend functionality via REST or GraphQL API.
- **Continuous Crawling:** Scheduling periodic data/FAQ updates.
- **Enhanced Entity Linking:** Improved NLP for entity disambiguation and linking.

---