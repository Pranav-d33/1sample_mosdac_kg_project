# MOSDAC Knowledge Graph Project (Backend)

This repository implements a backend pipeline for building a knowledge graph and enabling hybrid question answering over the MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) portal and related sources. The system combines traditional knowledge graph methods, vector-based semantic search, and FAQ extraction to provide rich and accurate responses to user queries.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline & Script Order](#pipeline--script-order)
- [Script Descriptions](#script-descriptions)
- [Outputs](#outputs)
- [Setup & Usage](#setup--usage)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This backend processes web documents and FAQs from MOSDAC and related sources to:
- Build a knowledge graph (KG) of entities and their relationships.
- Create a semantic vector index for document retrieval.
- Extract and structure FAQs.
- Provide a hybrid answer generator that leverages KG, semantic retrieval, and FAQs.
- Expose all retrieval and answering functionality for integration with a frontend (to be added).

---

## Architecture

The system architecture can be visualized as follows:

![System Architecture](image1)

- **Data ingestion** from web and FAQ sources.
- **Text & entity extraction** from crawled documents.
- **Knowledge graph construction** for structured relations.
- **Vector index building** for semantic search.
- **Hybrid answer generator** combining KG, vector, and FAQ retrieval.
- **User queries** are handled via the hybrid answer engine.

---

## Pipeline & Script Order

Below is the recommended order for running the backend scripts:

1. **Data Acquisition**
   - `crawl_documents.py`: Crawl and collect document URLs.
   - `download_documents.py`: Download listed documents.
   - `crawl_faqs.py`: Crawl FAQ data from the MOSDAC FAQ page.

2. **Document Processing**
   - `extract_text_from_docs.py`: Extract raw text from downloaded documents.

3. **Entity Extraction & Knowledge Graph Construction**
   - `extract_entities.py`: Extract entities from document text.
   - `build_graph.py`: Build a knowledge graph from extracted entities.
   - `normalize_graph.py`: Normalize the graph for efficient retrieval.

4. **Indexing & Retrieval Preparation**
   - `vector_retriever.py`: Build a vector index for semantic search.
   - `query_retriever.py`: Retrieve relevant documents/entities for queries.

5. **Utilities**
   - `entity_extractor.py`: Utility for matching user queries to KG entities.
   - `graph_search.py`: Enables searching/traversal over the knowledge graph.
   - `prompt_builder.py`: Utility for constructing prompts and query formats.

6. **Hybrid Answer Generation**
   - `hybrid_answer_generator.py`: Main answer engine. Combines KG, FAQ, vector retrieval, and utilities to answer user queries.

---

## Script Descriptions

| Script                    | Description                                                                                   |
|---------------------------|----------------------------------------------------------------------------------------------|
| crawl_documents.py        | Crawls web sources for document links (MOSDAC, etc.).                                        |
| download_documents.py     | Downloads documents from the collected URLs.                                                 |
| crawl_faqs.py             | Crawls and extracts FAQ data from the MOSDAC FAQ page.                                       |
| extract_text_from_docs.py | Extracts text from downloaded documents.                                                     |
| extract_entities.py       | Performs NLP-based entity extraction from document text.                                     |
| build_graph.py            | Builds a knowledge graph from extracted entities.                                            |
| normalize_graph.py        | Normalizes the knowledge graph for efficient search and retrieval.                           |
| vector_retriever.py       | Builds a vector search index (e.g., using FAISS, SentenceTransformers) for semantic search.  |
| query_retriever.py        | Retrieves relevant documents/entities from the index for user queries.                       |
| entity_extractor.py       | Extracts/matches entities from user queries to KG nodes.                                     |
| graph_search.py           | Enables advanced search/traversal operations on the knowledge graph.                         |
| prompt_builder.py         | Utility for constructing prompts and query formats for answer generation.                    |
| hybrid_answer_generator.py| The main answer engine—combines KG, vector search, and FAQ data for hybrid Q&A responses.    |

---

## Outputs

The pipeline produces the following key outputs:

- `outputs/cleaned_json/documents.json` — Collected document URLs
- `outputs/cleaned_json/cleaned_docs.json` — Cleaned document text
- `outputs/cleaned_json/faqs.json` — Extracted FAQ data
- `outputs/phase2/entities.json` — Extracted entities
- `outputs/phase2/knowledge_graph.json` / `.graphml` — Raw knowledge graph
- `outputs/knowledge_graph_normalized.json` — Normalized knowledge graph
- Vector index files for semantic retrieval (output by `vector_retriever.py`)

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- Required Python libraries (see `requirements.txt` if provided)
- Network access to crawl MOSDAC and fetch documents

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Pranav-d33/1sample_mosdac_kg_project.git
   cd 1sample_mosdac_kg_project/scripts
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

### Running the Pipeline

Follow the [Pipeline & Script Order](#pipeline--script-order) section to run each script in sequence.

Example:
```sh
python crawl_documents.py
python download_documents.py
python crawl_faqs.py
python extract_text_from_docs.py
# ...continue as described above
```

**Note:** Some utility scripts (like `prompt_builder.py`, `entity_extractor.py`) are typically imported as modules and not run directly.

---

## Future Work

- **Frontend Integration:** A frontend interface will be added in future commits.
- **API/Service Layer:** Exposing backend functionality via REST or GraphQL API.
- **Continuous Crawling:** Scheduling periodic data/FAQ updates.
- **Enhanced Entity Linking:** Improved NLP for entity disambiguation and linking.

---

