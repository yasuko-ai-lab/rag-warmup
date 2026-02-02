# Changelog

All notable changes to this project are documented in this file.

This project follows a simple versioning scheme suitable for learning and demonstration purposes.

---

## [v1.1] – Multi-Document RAG, Metadata & Retrieval Controls (2026-01)

### Added
- Support for **multiple documents** (`.md` and `.txt`) loaded automatically from the `docs/` directory
- **Per-chunk metadata** attached during ingestion, including:
  - source filename
  - section title
  - chunk ID
- Source attribution displayed in the Streamlit UI for every answer
- Expandable **Sources** panel showing ranked retrieved chunks
- Distance-based retrieval filtering (`MAX_DISTANCE`)
  - Filters out weak semantic matches before context construction
  - Ensures the model answers only when sufficiently relevant sources are found
  - Results in explicit “I don’t know” responses when no reliable context exists

### Changed
- Refactored retriever from single-document to **multi-document ingestion pipeline**
- Chunking logic updated to operate per-file while preserving document boundaries
- Retrieval results now return structured records instead of raw text
- Streamlit UI enhanced to surface top source, section, and similarity distance

### Fixed
- Chunk count mismatches caused by inconsistent heading formats
- Improved handling of mixed line endings across document types
- Prevented top-level document titles from polluting retrieval results

### Notes
- This version prioritizes **explainability, traceability, and correctness** over performance
- Embeddings and FAISS index are cached to improve interactive performance

---

## [v1.0] – Initial RAG Warm-up Implementation (2026-01)

### Added
- Basic Retrieval-Augmented Generation (RAG) pipeline
- Single-document chunking and embedding
- FAISS vector index for similarity search
- Command-line based retrieval and answering
- Streamlit chat interface
- Hallucination-controlled prompt design
- Explicit fallback behavior when answers are not found in documents

### Notes
- Designed as a learning-focused warm-up project
- Emphasis on clarity and correctness rather than scale or optimization

