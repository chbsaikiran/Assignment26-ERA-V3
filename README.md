# Semantic Search Web Article System

This project implements a semantic search system that processes web articles and allows users to search through their content using natural language queries. The system uses FAISS for similarity search, Google's Gemini API for embeddings, and provides a web interface for searching and viewing results.

## Features

- Web article content extraction and processing
- Intelligent text chunking based on context
- Semantic search using FAISS and Gemini embeddings
- Interactive web interface with highlighted search results
- Real-time search functionality
- Automatic scrolling to matched content

## Prerequisites

- Python 3.10+
- Google Cloud account with Gemini API access
- Environment variables set up (see Configuration section)

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

2. Ensure the following directories exist (they will be created automatically if missing):
- `uploaded_files/`
- `static/`
- `documents/`

## Project Structure

- `main.py`: FastAPI server implementation with search endpoints
- `get_the_metadata_and_index.py`: Core functionality for processing articles and creating search indices
- `static/index.html`: Web interface for the search system
- `requirements.txt`: Project dependencies

## How It Works

1. **Article Processing (`get_the_metadata_and_index.py`)**:
   - Downloads web article content using trafilatura
   - Splits content into meaningful chunks using Gemini AI
   - Generates embeddings for each chunk using Gemini's embedding model
   - Creates a FAISS index for efficient similarity search
   - Stores metadata about chunks and their sources

2. **Search Server (`main.py`)**:
   - Provides REST API endpoints for search functionality
   - Processes search queries and returns relevant results
   - Handles text highlighting and result display

3. **Web Interface (`index.html`)**:
   - Provides a user-friendly search interface
   - Displays search results with highlighted matches
   - Automatically scrolls to relevant content

## Running the System

1. First, process the articles and create the search index:
```bash
python get_the_metadata_and_index.py
```

2. Start the FastAPI server:
```bash
python main.py
```

3. Access the web interface at `http://localhost:8000`

## Usage

1. Enter a search query in the search box
2. Click "Search" or press Enter
3. The system will:
   - Find the most semantically relevant content
   - Display the article with the matched text highlighted
   - Automatically scroll to the highlighted section

## Technical Details

- **Embedding Model**: Uses Gemini's embedding-001 model
- **Search Index**: FAISS IndexFlatL2 for efficient similarity search
- **Text Processing**: Intelligent chunking with context-aware splits
- **Web Framework**: FastAPI with CORS support
- **Frontend**: Pure HTML/JavaScript with no external dependencies

## Current Limitations

- Processes a fixed set of articles defined in the code
- Requires manual addition of new articles
- Limited to text content (image processing is commented out)

## Future Improvements

- Add support for dynamic article addition
- Implement image processing and search
- Add pagination for search results
- Improve chunk size and overlap parameters
- Add support for multiple search results
