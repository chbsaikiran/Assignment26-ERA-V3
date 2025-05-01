import os
from pathlib import Path
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
print("hello0")

# -- CONFIG --
CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
DOC_PATH = Path("documents")

# -- HELPERS --

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embedding(text: str) -> np.ndarray:
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(embedding['embedding'], dtype=np.float32)

# -- LOAD DOCS & CHUNK --
all_chunks = []
metadata = []

for file in DOC_PATH.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(get_embedding(chunk))
            metadata.append({
                "doc_name": file.name,
                "chunk": chunk,
                "chunk_id": f"{file.stem}_{idx}"
            })
    print("Sleeping")
    time.sleep(20)

# -- CREATE FAISS INDEX --
dimension = len(all_chunks[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.stack(all_chunks))

print(f"✅ Indexed {len(all_chunks)} chunks from {len(list(DOC_PATH.glob('*.txt')))} documents")

# -- SEARCH --
query = "When will Dhoni retire?"
query_vec = get_embedding(query).reshape(1, -1)
D, I = index.search(query_vec, k=3)

print(f"\n🔍 Query: {query}\n\n📚 Top Matches:")
for rank, idx in enumerate(I[0]):
    data = metadata[idx]
    print(f"\n#{rank + 1}: From {data['doc_name']} [{data['chunk_id']}]")
    print(f"→ {data['chunk']}")
