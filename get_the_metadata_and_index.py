import trafilatura
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
import faiss

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure the model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# List available models
# print("Available Models:")
# for m in genai.list_models():
#     print(f"- {m.name}")

# Initialize the model
model = genai.GenerativeModel("gemini-2.0-flash",
                            generation_config=generation_config)

def extract_text_and_images(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            print(f"Failed to download content from URL: {url}")
            return None, None

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text is None:
            print(f"Failed to extract text from URL: {url}")
            return None, None

        # Extract images
        soup = BeautifulSoup(downloaded, 'html.parser')
        img_urls = []
        for img_tag in soup.find_all('img'):
            src = img_tag.get('src')
            if src:
                if src.startswith('//'):
                    src = 'https:' + src
                elif not src.startswith(('http://', 'https://')):
                    src = urljoin(url, src)
                img_urls.append(src)

        return text, img_urls
    except Exception as e:
        print(f"Error in extract_text_and_images: {str(e)}")
        return None, None

def generate_metadata(url, model):
    try:
        text, img_urls = extract_text_and_images(url)
        if not text:
            print("No text found in the URL.")
            return None

        # Generate metadata for text chunks
        metadata, idx = intelligent_chunking(text, url, model)

        # Add image entries after text chunks
        # if img_urls:
        #     for img_url in img_urls:
        #         metadata.append({
        #             "doc_name": url,
        #             "chunk": f"[IMAGE] {img_url}",
        #             "chunk_id": f"{url}_{idx}"
        #         })
        #         idx += 1

        # Save metadata to file
        output_file = "metadata.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Metadata saved to {output_file} with {len(metadata)} entries.")
        return metadata
    except Exception as e:
        print(f"Error in generate_metadata: {str(e)}")
        return None

def find_context_split(text_128, model):
    """Ask Gemini where the context shifts in a 128-word text segment."""
    try:
        prompt = (
            "You will be given a 128-word text. "
            "Tell me after how many words (integer) a clear context or topic shift occurs. "
            "If no shift, return 128. Only return the integer.\n\n"
            f"{text_128}"
        )
        response = model.generate_content(prompt)
        # Extract just the number from the response
        try:
            split_index = int(''.join(filter(str.isdigit, response.text.strip())))
            return min(max(split_index, 1), 128)  # Ensure result is between 1 and 128
        except ValueError:
            print("Could not parse integer from model response")
            return 128
    except Exception as e:
        print(f"Error in find_context_split: {str(e)}")
        return 128  # fallback to maximum length

def intelligent_chunking(full_text, url, model):
    """Split text into chunks based on context shifts."""
    try:
        words = full_text.split()
        pointer = 0
        buffer = []
        metadata = []
        idx = 0

        while pointer < len(words):
            # Form 128-word window
            window_size = 128 - len(buffer)
            segment = buffer + words[pointer:pointer + window_size]
            if not segment:
                break

            # Get split point from model
            text_128 = ' '.join(segment[:128])  # Ensure we don't exceed 128 words
            split_idx = find_context_split(text_128, model)

            # Create chunk and add to metadata
            chunk = ' '.join(segment[:split_idx])
            if chunk.strip():  # Only add non-empty chunks
                metadata.append({
                    "doc_name": url,
                    "chunk": chunk,
                    "chunk_id": f"{url}_{idx}"
                })
                idx += 1

            # Update buffer and pointer
            buffer = segment[split_idx:]
            pointer += window_size

        return metadata, idx
    except Exception as e:
        print(f"Error in intelligent_chunking: {str(e)}")
        return [], 0

def get_embedding(text: str) -> np.ndarray:
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(embedding['embedding'], dtype=np.float32)

def main():
    try:
        # Example URL
        url = "https://economictimes.indiatimes.com/tech/catalysts/how-data-ai-tech-and-quantum-computing-are-shaping-the-future-of-work/articleshow/111078862.cms"
        
        # Generate metadata
        metadata = generate_metadata(url, model)
        #with open("metadata.json", "w", encoding="utf-8") as f:
        #    json.dump(metadata, f, indent=2, ensure_ascii=False)

        if metadata:
            print("Processing completed successfully!")
        else:
            print("Failed to process the URL.")

        text_chunks = [entry['chunk'] for entry in metadata if not entry['chunk'].startswith('[IMAGE]')]

        all_chunks = []

        for chunk in text_chunks:
            all_chunks.append(get_embedding(chunk))
        # Build FAISS index
        dimension = len(all_chunks[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.stack(all_chunks))
        # Save FAISS index
        faiss.write_index(index, "chunk_index.faiss")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

