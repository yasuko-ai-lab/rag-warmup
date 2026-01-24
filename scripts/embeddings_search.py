import os
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI

# --- API key check ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "❌ OPENAI_API_KEY not set.\n"
        "Run:\n"
        "export OPENAI_API_KEY='your-key-here'"
    )

client = OpenAI(api_key=api_key)

# --- Load document ---
text = Path("docs/ai_notes.md").read_text(encoding="utf-8")

# --- Chunking ---
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

chunks = []
start = 0
##while start < len(text):
##    end = start + CHUNK_SIZE
##    chunks.append(text[start:end])
##    start = end - CHUNK_OVERLAP

## Alternative chunking by splitting on headings
parts = text.split("## ")
for i, p in enumerate(parts):
    if i == 0:
        chunks.append(p.strip())
    else:
        chunks.append(("## " + p).strip())

print(f"Chunks: {len(chunks)}\n")
for i, c in enumerate(chunks, start=1):
    print(f"--- Chunk {i} ---\n{c}\n")

# --- Embedding function ---
def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# --- Embed chunks ---
embeddings = [embed(chunk) for chunk in chunks]
X = np.array(embeddings, dtype="float32")

# --- FAISS index ---
dimension = X.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(X)

print(f"✅ FAISS index built with {index.ntotal} vectors")

# --- Search ---
query = "What is RAG?"
query_embedding = np.array([embed(query)], dtype="float32")

k = 2
distances, indices = index.search(query_embedding, k)

print("\n🔍 Query:", query)
print("Top results:\n")

for rank, idx in enumerate(indices[0], start=1):
    print(f"--- Result {rank} (distance={distances[0][rank-1]:.4f}) ---")
    print(chunks[idx].strip())
    print()
