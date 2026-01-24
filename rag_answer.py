import os
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY first: export OPENAI_API_KEY='...'")

client = OpenAI(api_key=api_key)

# ---- Load + chunk (markdown headings) ----
text = Path("docs/ai_notes.md").read_text(encoding="utf-8")

parts = text.split("\n## ")
chunks = []
for i, p in enumerate(parts):
    p = p.strip()
    if not p:
        continue
    chunks.append(p if i == 0 else "## " + p)

# Optional: drop the top title chunk if it's too generic
chunks = [c for c in chunks if not c.startswith("# ")]

def embed(t: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=t)
    return resp.data[0].embedding

# ---- Build FAISS ----
X = np.array([embed(c) for c in chunks], dtype="float32")
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)

def retrieve(query: str, k: int = 3):
    q = np.array([embed(query)], dtype="float32")
    distances, indices = index.search(q, k)
    results = []
    for rank, idx in enumerate(indices[0], start=1):
        results.append({"rank": rank, "chunk": chunks[idx], "distance": float(distances[0][rank-1])})
    return results

def answer(query: str, k: int = 3) -> dict:
    results = retrieve(query, k=k)
    context = "\n\n---\n\n".join([r["chunk"] for r in results])

    prompt = f"""Answer the user's question using ONLY the context.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{query}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only uses the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": results
    }

if __name__ == "__main__":
    q = input("Ask a question: ").strip()
    out = answer(q, k=3)
    print("\n✅ Answer:\n")
    print(out["answer"])
    print("\n📌 Sources used:\n")
    for s in out["sources"]:
        print(f"- Rank {s['rank']} (distance={s['distance']:.4f})")
        print(s["chunk"])
        print()
