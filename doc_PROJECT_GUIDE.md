# RAG for Book Recommendation - Project Guide

This guide explains how to run and understand the RAG (Retrieval-Augmented Generation) Book Recommendation project, based on the code in L2.ipynb.

## Project Overview
- Reads book descriptions from text files in `include/data/`.
- Uses `fastembed` to create vector embeddings for each book description.
- Stores embeddings and metadata in a local Weaviate vector database (embedded mode).
- Allows semantic search for book recommendations.

## Project Structure
- `rag_book_recommendation.py`: Main script with all logic from the notebook.
- `requirements.txt`: Python dependencies.
- `helper.py`: Utility functions (e.g., `suppress_output`).
- `include/data/`: Folder with book description `.txt` files.
- `Dockerfile`: For containerized execution.

## Step-by-Step Instructions

### 1. Prepare Data
- Place your book description files in `include/data/`.
- Each line in a file should be:
  `[Index] ::: [Book Title] ([Year]) ::: [Author] ::: [Description]`

### 2. Install Requirements (Locally)
```bash
pip install -r requirements.txt
```

### 3. Run the Project (Locally)
```bash
python rag_book_recommendation.py
```

### 4. Run with Docker
Build the Docker image:
```bash
docker build -t rag-book-recommendation .
```
Run the container:
```bash
docker run --rm -v %cd%/include/data:/app/include/data rag-book-recommendation
```
(For Linux/Mac, use `$(pwd)` instead of `%cd%`.)

### 5. What the Script Does
- Starts an embedded Weaviate instance.
- Creates a collection for books if it doesn't exist.
- Reads all `.txt` files in `include/data/` and parses book info.
- Generates embeddings for each book description.
- Loads all books and their embeddings into Weaviate.
- Runs a sample semantic search: recommends a book for the query "A philosophical book".

### 6. Customization
- Add your own `.txt` files to `include/data/` with the same format.
- Change the query string in `rag_book_recommendation.py` to test different recommendations.

### 7. Cleanup (Optional)
- Remove a book file: delete it from `include/data/`.
- Remove the Weaviate data: delete the `tmp/weaviate` directory.

### 8. Resources
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [FastEmbed Docs](https://qdrant.github.io/fastembed/)

---
For more details, see the comments in `rag_book_recommendation.py` and the original notebook.
