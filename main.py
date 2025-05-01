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
    latest_content = {"type": "text", "content": text_data.text}
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