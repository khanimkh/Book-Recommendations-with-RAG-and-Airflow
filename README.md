# Book Recommendations with RAG + Airflow

An end-to-end Book Recommendation System that combines Retrieval-Augmented Generation (RAG), Apache Airflow orchestration, Weaviate vector search, and a FastAPI web application.

## What This Project Does

This project ingests book records from text files and manual user input, converts descriptions into embeddings, stores vectors and metadata in Weaviate, and returns recommendations for user queries. It also includes book management features (add, upload, edit, delete) and a chatbot interface for recommendation-style Q&A.

## High-Level Architecture

The runtime stack is managed with Docker Compose and includes:

- Airflow services (scheduler, worker, triggerer, API server, dag processor)
- PostgreSQL (Airflow metadata database)
- Redis (Celery broker)
- Weaviate (vector database)
- text2vec-transformers (vectorizer module service)
- Ollama (LLM serving for chatbot responses)
- FastAPI web app (Book UI)

Main orchestration and app flow:

1. Airflow DAG reads data files and normalizes records.
2. Embeddings are generated and loaded into Weaviate.
3. Web app supports search, chat, and CRUD-style book operations.
4. Web app triggers Airflow DAGs for background synchronization.

## Project Structure

- Airflow/: Main runtime folder for Dockerized services and application code
- Airflow/docker-compose.yaml: Service definitions, environment variables, ports, and volumes
- Airflow/Dockerfile: Custom Airflow image build
- Airflow/requirements.txt: Airflow container Python dependencies
- Airflow/dags/fetch_data.py: Batch pipeline (discover files -> transform -> embed -> load)
- Airflow/dags/add_book_data.py: Single-book insertion DAG
- Airflow/dags/query_data.py: Retrieval/logging DAG
- Airflow/webapp/main.py: FastAPI routes for upload/add/search/chat/edit/delete
- Airflow/webapp/templates/index.html: Web UI template
- Airflow/include/data/: Book input files (mounted in containers)
- Airflow/logs/: Airflow task logs
- doc_BOOK_RECOMMENDATION_RAG_AIRFLOW_TEACHING.md: Deep teaching guide

## Data Format

Each input line in .txt files should follow:

id:::title:::year:::publisher:::category:::description

Notes:

- Category can be empty and is auto-inferred by keyword heuristics.
- Duplicate checks use title + year + publisher as a logical key.

## Quick Start

Prerequisites:

- Docker Desktop (or Docker Engine + Docker Compose)

Run:

```powershell
cd "Airflow"
docker compose up -d
```

Access:

- Airflow UI: http://localhost:8080
- Book Web UI: http://localhost:8090
- Weaviate API (host): http://localhost:8081

Default Airflow user/password are typically:

- Username: airflow
- Password: airflow

## Core Pipeline (Airflow DAGs)

### fetch_data

Purpose:

- List all .txt files from include/data
- Transform lines into structured records
- Generate embeddings
- Insert into Weaviate

Characteristics:

- Scheduled hourly
- Includes schema checks and duplicate protection
- Uses dynamic task mapping for per-file processing

### add_book_data

Purpose:

- Handle single-book insertion triggered by web app
- Generate embedding and insert immediately

### query_data

Purpose:

- Run retrieval/search task from query payload
- Can run on data-aware triggers (asset updates)

## Web App Features

Main user operations in the FastAPI app:

- Upload books file (.txt)
- Add a book manually by fields
- Search books by title/year/publisher/category/context
- Edit existing book entries
- Delete books
- Chatbot endpoint for recommendation-style responses

Implementation highlights:

- Hybrid retrieval (vector + BM25) when embeddings are available
- Lexical fallback for resilience when embedding/LLM services are unavailable
- Best-effort DAG triggers for non-blocking UI interactions

## Logging and Error Handling

- Airflow task logs are available in Airflow/logs and Airflow UI.
- DAGs include retry policies and schema compatibility checks.
- Web app routes catch exceptions and preserve usability with fallbacks.
- Data file updates are synchronized with edit/delete actions to avoid re-ingesting stale records.

## Common Commands

From Airflow/:

```powershell
# Check container status
docker compose ps

# Follow logs for key services
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-worker
docker compose logs -f book-ui
docker compose logs -f weaviate

# Stop everything
docker compose down
```

## Documentation

- Teaching Guide: [doc_BOOK_RECOMMENDATION_RAG_AIRFLOW_TEACHING.md](doc_BOOK_RECOMMENDATION_RAG_AIRFLOW_TEACHING.md)
- Apache Airflow Docs: https://airflow.apache.org/docs/
- Airflow Docker Compose Guide: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
- Weaviate Docs: https://weaviate.io/developers/weaviate
- FastAPI Docs: https://fastapi.tiangolo.com/
- Ollama: https://github.com/ollama/ollama
