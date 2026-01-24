from pathlib import Path

# Load document
doc_path = Path("docs/ai_notes.md")
text = doc_path.read_text(encoding="utf-8")

# Simple chunking parameters
CHUNK_SIZE = 300 # characters
CHUNK_OVERLAP = 50

chunks = []
start = 0

while start < len(text):
  end = start + CHUNK_SIZE
  chunk = text[start:end]
  chunks.append(chunk)
  start = end - CHUNK_OVERLAP

# Print results
print(f"Total chuncks created: {len(chunks)}\n")

for i, chunk in enumerate(chunks, start=2):
  print(f"---Chunk {i} ---")
  print(chunk)
  print()
