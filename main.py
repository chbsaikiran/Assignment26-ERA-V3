from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
from pathlib import Path
import os
import json
import requests
import re
from bs4 import BeautifulSoup

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create directories for uploads and static files
UPLOAD_DIR = Path("uploaded_files")
STATIC_DIR = Path("static")
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Store the latest content
latest_content = {"type": None, "content": None}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

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
        
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the first occurrence of the text
        text_to_find = best_match['chunk'].replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
        pattern = re.compile(f'({text_to_find})', re.IGNORECASE)
        
        # Function to wrap text with highlight
        def highlight_match(text):
            return f'<mark style="background-color: yellow;">{text.group(1)}</mark>'
        
        # Replace first occurrence with highlighted version
        modified_html = pattern.sub(highlight_match, str(soup), count=1)
        
        return HTMLResponse(content=modified_html, status_code=200)
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