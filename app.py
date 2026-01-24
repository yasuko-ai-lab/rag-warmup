import os
from pathlib import Path

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Mini RAG Chat", layout="centered")
st.title("📄 Mini RAG Chat")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set. In your terminal run:\n\nexport OPENAI_API_KEY='your-key-here'")
    st.stop()

client = OpenAI(api_key=api_key)

@st.cache_resource
def build_retriever():
    # Load + chunk markdown by headings
    text = Path("docs/ai_notes.md").read_text(encoding="utf-8")

    parts = text.split("\n## ")
    chunks = []
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        chunks.append(p if i == 0 else "## " + p)

    # Drop top-level title chunk (usually too generic)
    chunks = [c for c in chunks if not c.startswith("# ")]

    def embed(t: str) -> list[float]:
        r = client.embeddings.create(model="text-embedding-3-small", input=t)
        return r.data[0].embedding

    # Build FAISS index
    X = np.array([embed(c) for c in chunks], dtype="float32")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    return index, chunks, embed

index, chunks, embed = build_retriever()

def retrieve(query: str, k: int = 3):
    q = np.array([embed(query)], dtype="float32")
    distances, indices = index.search(q, k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        results.append({
            "rank": rank,
            "distance": float(distances[0][rank - 1]),
            "chunk": chunks[idx],
        })
    return results

def rag_answer(query: str, k: int = 3):
    results = retrieve(query, k=k)
    context = "\n\n---\n\n".join([r["chunk"] for r in results])

    prompt = f"""Answer the user's question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{query}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You only use the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content, results

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**Rank {s['rank']} (distance={s['distance']:.4f})**")
                    st.code(s["chunk"])

# New input
user_q = st.chat_input("Ask a question about your docs…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        answer, sources = rag_answer(user_q, k=3)
        st.markdown(answer)

        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**Rank {s['rank']} (distance={s['distance']:.4f})**")
                st.code(s["chunk"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
