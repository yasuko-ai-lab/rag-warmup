import os
from pathlib import Path

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI
import re    

st.set_page_config(page_title="Mini RAG Chat", layout="centered")
st.title("📄 Mini RAG Chat")

if st.button("Reset chat"):
    st.session_state.messages = []
    st.rerun()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set. In your terminal run:\n\nexport OPENAI_API_KEY='your-key-here'")
    st.stop()

client = OpenAI(api_key=api_key)

@st.cache_resource
def build_retriever():
    
    #multiple documents support
    docs_dir = Path("docs")
    
    paths = sorted(list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))) 
    
    if not paths:
        raise SystemExit("No .md/.txt files found in docs/")

    chunk_records = []
    chunk_id = 0    


    for path in paths:
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        
        parts = raw.split("\n## ")
        for i, part in enumerate(parts):
          part = part.strip()
          if not part:
              continue
          
          text = part if i == 0 else "## " + part
 
          # Section = heading text (best-effort)
          if text.startswith("## "):
              section = text.splitlines()[0].replace("## ", "").strip()
          else:
              section = "(intro)"

          # Drop top-level titles if they appear
          if text.startswith("# "):
              continue

          chunk_records.append({
              "id": chunk_id,
              "source": path.name,
              "section": section,
              "text": text,
          })
          chunk_id += 1


    def embed(t: str) -> list[float]:
        r = client.embeddings.create(model="text-embedding-3-small", input=t)
        return r.data[0].embedding

    # Build FAISS index
    X = np.array([embed(c["text"]) for c in chunk_records], dtype="float32")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    return index, chunk_records, embed

index, chunks, embed = build_retriever()

#st.write("Total chucnks:", len(chunks))

def retrieve(query: str, k: int = 3):
    q = np.array([embed(query)], dtype="float32")
    distances, indices = index.search(q, k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        c = chunks[idx]
        results.append({
            "rank": rank,
            "distance": float(distances[0][rank - 1]),
            "chunk": c["text"],
            "source": c["source"],
            "section": c["section"],
            "chunk_id": c["id"]
        })
    return results

def rag_answer(query: str, k: int = 3):
    results = retrieve(query, k=k)
    
    MAX_DISTANCE = 0.6  # start here; tune later
    results = [r for r in results if r["distance"] <= MAX_DISTANCE]
    
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
user_q = st.chat_input("Ask a question about your docs…(e.g. “What is RAG?” or “Define LLM”)")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        answer, sources = rag_answer(user_q, k=3)

        if sources:
            top = sources[0]
            st.caption(
                f"Top source: {top['source']} → {top['section']} "
                f"(distance={top['distance']:.4f})"
            )
        else:
            st.caption("Top source: (none)")

        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(
                        f"**{s['source']} → {s['section']}** "
                        f"(Rank {s['rank']}, distance={s['distance']:.4f})"
                    )
                    st.code(s["chunk"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
