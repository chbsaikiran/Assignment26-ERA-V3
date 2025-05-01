from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
from pathlib import Path
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a directory to store uploaded images
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

import os
from pathlib import Path
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#print("hello0")

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

def get_indices(all_chunks, metadata):
    # -- LOAD DOCS & CHUNK --
    
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
        #print("Sleeping")
        time.sleep(2)
    
    # -- CREATE FAISS INDEX --
    dimension = len(all_chunks[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.stack(all_chunks))
    return index
    
    #print(f"✅ Indexed {len(all_chunks)} chunks from {len(list(DOC_PATH.glob('*.txt')))} documents")

class TextData(BaseModel):
    text: str

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>FastAPI Server</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .content { max-width: 800px; margin: 0 auto; }
                img { max-width: 100%; }
                .message { padding: 20px; background: #f0f0f0; margin: 20px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="content">
                <h1>FastAPI Server</h1>
                <div id="content">
                    <p>Waiting for content from Chrome extension...</p>
                </div>
            </div>
            <script>
                function checkForUpdates() {
                    fetch('/get_latest')
                        .then(response => response.json())
                        .then(data => {
                            const contentDiv = document.getElementById('content');
                            if (data.type === 'text') {
                                contentDiv.innerHTML = `<div class="message">${data.content}</div>`;
                            } else if (data.type === 'image') {
                                contentDiv.innerHTML = `<img src="${data.content}" alt="Uploaded image">`;
                            }
                        });
                }
                setInterval(checkForUpdates, 1000);
            </script>
        </body>
    </html>
    """)

# Store the latest content
latest_content = {"type": None, "content": None}

@app.get("/get_latest")
async def get_latest():
    return latest_content

@app.post("/process_text")
async def process_text(text_data: TextData):
    global latest_content
    query = text_data.text
    query_vec = get_embedding(query).reshape(1, -1)
    all_chunks = []
    metadata = []
    index = get_indices(all_chunks, metadata)
    D, I = index.search(query_vec, k=3)
    results = ""
    #print(f"\n🔍 Query: {query}\n\n📚 Top Matches:")
    for rank, idx in enumerate(I[0]):
        data = metadata[idx]
        results += f"\n#{rank + 1}: From {data['doc_name']} [{data['chunk_id']}]\n"
        results += f"\n→ {data['chunk']}\n\n"
    latest_content = {"type": "text", "content": results}
    return {"message": "Text received successfully"}

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    global latest_content
    
    # Read the file content
    content = await file.read()
    
    # Convert to base64 for displaying in HTML
    base64_image = base64.b64encode(content).decode()
    image_src = f"data:{file.content_type};base64,{base64_image}"
    
    # Save the image
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(content)
    
    latest_content = {"type": "image", "content": image_src}
    return {"message": "Image received successfully"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 