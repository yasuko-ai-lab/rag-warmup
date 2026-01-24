from pathlib import Path

#Path to the markdown file
doc_path = Path("docs/ai_notes.md")

#Read the file
text = doc_path.read_text(encoding="utf-8")

print("✅ Document loaded successfully!\n")
print(text)
