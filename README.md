# Mini RAG App

A small Retrieval-Augmented Generation (RAG) app that answers questions using custom markdown documents instead of relying only on a model’s pre-trained knowledge.

## What it does
- Loads markdown notes from `docs/`
- Chunks documents by section headings
- Generates embeddings using OpenAI embeddings
- Stores vectors in FAISS for semantic search
- Retrieves top-k relevant chunks per query
- Generates answers using only retrieved context
- Streamlit chat UI + shows source chunks

## Project structure
```text
rag-warmup/
├── app.py            # Streamlit UI
├── rag_answer.py     # CLI version (retrieval + generation)
├── docs/             # Knowledge base (markdown)
├── scripts/          # Learning / dev scripts
├── requirements.txt
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

Notes

If the answer is not found in the documents, the system responds with “I don’t know based on the provided documents.”

Future improvements

Support multiple documents and PDFs

Persist FAISS index to disk

Add metadata (filename, section titles)

Add relevance thresholds or reranking
