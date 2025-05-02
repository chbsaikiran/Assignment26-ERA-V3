from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
from pathlib import Path
import os
import json

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
                .highlight { background-color: yellow; }
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
                    console.log("Checking for updates...");  // Debug log
                    fetch('/get_latest')
                        .then(response => response.json())
                        .then(data => {
                            console.log("Received data:", data);  // Debug log
                            const contentDiv = document.getElementById('content');
                            if (data.type === 'text' && data.content) {
                                console.log("Updating content with:", data.content);  // Debug log
                                contentDiv.innerHTML = data.content;
                            } else if (data.type === 'image') {
                                contentDiv.innerHTML = `<img src="${data.content}" alt="Uploaded image">`;
                            }
                        })
                        .catch(error => {
                            console.error('Error checking for updates:', error);
                        });
                }
                
                // Check for updates every second
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
    try:
        print("Received text:", text_data.text)
        query = text_data.text
        index = faiss.read_index("chunk_index.faiss")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        query_vec = get_embedding(query).reshape(1, -1)
        D, I = index.search(query_vec, k=1)
        
        best_match = metadata[I[0][0]]
        print("Found match:", best_match)
        
        # Ensure URL has proper format
        url = best_match['doc_name']
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Escape the text for JavaScript
        text_to_highlight = best_match['chunk'].replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
        
        html_content = f"""
        <div id="result-container">
            <h3>Found matching content!</h3>
            <div>
                <strong>URL:</strong> <a href="{url}" target="_blank">{url}</a>
            </div>
            <div>
                <strong>Text to highlight:</strong>
                <div style="background: #f5f5f5; padding: 10px; margin: 10px 0;">
                    {text_to_highlight}
                </div>
            </div>
            <button onclick="window.open('{url}', '_blank')">Open URL in New Tab</button>
        </div>
        """
        
        latest_content = {
            "type": "text",
            "content": html_content
        }
        return {"message": "Text received successfully"}
    except Exception as e:
        print(f"Error in process_text: {str(e)}")
        return {"message": f"Error processing text: {str(e)}"}

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