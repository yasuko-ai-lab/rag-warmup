# Mini RAG App

A multi-document **Retrieval-Augmented Generation (RAG)** demo that answers questions using a custom document corpus instead of relying solely on a model’s pre-trained knowledge.

This project focuses on **explainability, traceability, and hallucination control**, and is designed as a learning-focused but production-minded RAG implementation.

---

## What it does

- Loads **multiple documents** (`.md` and `.txt`) from the `docs/` directory
- Chunks documents by second-level Markdown headings (`##`)
- Generates embeddings using OpenAI embeddings
- Stores vectors in FAISS for semantic similarity search
- Retrieves top-k relevant chunks per query
- Applies **distance-based filtering** to discard weak matches
- Generates answers using **only retrieved context**
- Explicitly abstains when answers are not found in documents
- Streamlit chat UI with **source attribution and metadata**

---

## Key features

- Multi-document RAG ingestion
- Semantic chunking by document sections
- FAISS-based vector search
- Per-chunk metadata:
  - source filename
  - section title
  - chunk ID
- Distance-based retrieval filtering (`MAX_DISTANCE`)
- Hallucination-controlled prompt design
- Explainable answers with ranked source chunks
- Interactive Streamlit chat interface

---

## Project structure

```text
rag-warmup/
├── app.py            # Streamlit chat UI
├── rag_answer.py     # CLI version (retrieval + generation)
├── docs/             # Knowledge base (.md, .txt)
├── scripts/          # Learning / development scripts
├── requirements.txt
├── CHANGELOG.md
└── README.md

Setup
Install dependencies
pip install -r requirements.txt

Set API key
export OPENAI_API_KEY="your-api-key-here"

Run
Streamlit UI
streamlit run app.py

CLI version
python rag_answer.py

Behavior notes

If no retrieved chunk meets the similarity threshold, the system responds with:

“I don’t know based on the provided documents.”

Retrieved sources are displayed alongside each answer for transparency.

Versioning & changes

See CHANGELOG.md
 for a detailed history of improvements and design decisions.
