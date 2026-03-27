# Book Recommendation System with RAG and Airflow - Teaching Guide

## 1) What This Project Does
This project is an end-to-end Book Recommendation System that combines Retrieval-Augmented Generation (RAG), Apache Airflow orchestration, Weaviate vector search, and a FastAPI web interface. The system accepts book records (from text files and manual input), normalizes their metadata, generates vector embeddings for semantic retrieval, and stores both metadata and vectors in Weaviate. Users can then search by structured fields (title, year, publisher, category), by free-text context, and via a chatbot. The chatbot retrieves top relevant books and uses Ollama to generate friendly recommendation text grounded in retrieved records. Airflow DAGs orchestrate ingestion and retrieval tasks in a repeatable and observable way.

## 2) High-Level Architecture
At runtime, Docker Compose starts PostgreSQL, Redis, Airflow services, Weaviate, text2vec-transformers, Ollama, and the book web UI. Data ingestion and synchronization are orchestrated by DAGs in Airflow: one DAG (`fetch_data`) scans text files, transforms records, creates embeddings, and loads them to Weaviate; another (`add_book_data`) handles single-book DAG-triggered insertion; and another (`query_data`) performs retrieval/logging runs. The web app acts as the user-facing layer: it can upload files, add/edit/delete books, search, and chat. It also calls Airflow API to trigger DAGs in the background.

## 3) Project Structure
The core project folders and files are organized by responsibility:

- `Airflow/docker-compose.yaml`: service orchestration, networking, volumes, environment variables, and ports.
- `Airflow/Dockerfile`: custom Airflow image setup and dependency installation.
- `Airflow/requirements.txt`: Python dependencies for Airflow containers.
- `Airflow/dags/fetch_data.py`: batch ingestion pipeline (file discovery -> transform -> embedding -> load to Weaviate).
- `Airflow/dags/add_book_data.py`: single-book ingestion DAG triggered from the web app.
- `Airflow/dags/query_data.py`: retrieval DAG for query-time lookup/logging and asset-triggered runs.
- `Airflow/webapp/main.py`: FastAPI server with routes for upload/add/search/chat/edit/delete.
- `Airflow/webapp/templates/index.html`: Jinja2 template for the full UI.
- `Airflow/include/data/`: mounted directory containing book text files and manual book entries.
- `Airflow/logs/`: Airflow task logs by DAG run and task instance.

## 4) Libraries and Why They Are Used
This project uses a practical stack of libraries and services:

- FastAPI: HTTP API and server-side rendered pages.
- Jinja2Templates: HTML rendering for forms, messages, lists, and management views.
- requests: calling Airflow REST API (`/api/v1` with fallback `/api/v2`) to trigger DAG runs.
- weaviate-client: connection, schema operations, CRUD, hybrid/BM25/near-vector retrieval.
- fastembed: local embedding generation (`BAAI/bge-small-en-v1.5`) for semantic matching.
- ollama Python client: local LLM chat generation for recommendation responses.
- Apache Airflow: workflow orchestration, task retries, scheduling, and observability.
- PostgreSQL and Redis: Airflow metadata DB and Celery broker/result backend.
- Docker Compose: reproducible local deployment of all services.

## 5) Step-by-Step: Prepare Data
The input format expected per line is:

`id:::title:::year:::publisher:::category:::description`

Each line is parsed into a structured record with title, year, publisher, category, and description. If category is empty, category inference runs using keyword heuristics (for example, words like "stoic" or "wisdom" can map to Philosophy). During upload and add operations, duplicate checks are performed using a logical key `(title, year, publisher)`. This design ensures that manual and file-based ingestion can co-exist without repeated records. Data files are read from `Airflow/include/data`, which is mounted into containers as `/opt/airflow/include/data`.

## 6) Step-by-Step: Setup and Run the Project
To run the full system, start Docker Desktop (or Docker Engine), then run Docker Compose from the `Airflow` directory. The compose file builds a custom Airflow image and starts all dependent services. Airflow UI is available on port `8080`, Weaviate is exposed on `8081` externally (`8080` internally), and the web UI is on `8090`. The first startup may take longer because images, Python packages, and model artifacts need to be downloaded. Once containers are healthy, open the web UI and start ingesting/searching books.

Example command:

```powershell
cd "...\Book-Recommendations-with-RAG-Airflow\Airflow"
docker compose up -d
```

## 7) Step-by-Step: Read Book Descriptions and Ingest to Weaviate
The `fetch_data` DAG performs the batch pipeline. It first ensures the Weaviate collection exists with expected properties (`title`, `year`, `publisher`, `category`, `description`). Then it dynamically lists every `.txt` file in the data folder, transforms each line to normalized dictionaries, generates embeddings from description text, and inserts objects with vectors into Weaviate. This DAG also protects against schema drift by recreating the collection when old schemas are detected, and it avoids duplicate inserts by collecting existing logical keys before writing. Because the DAG is scheduled hourly and can also be triggered manually, it acts as both periodic synchronization and recovery pipeline.

Key code idea from `fetch_data.py`:

```python
book_description_files = [
    f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
]

_transform_book_description_files = transform_book_description_files.expand(
    book_description_file=_list_book_description_files
)

_create_vector_embeddings = create_vector_embeddings.expand(
    book_data=_transform_book_description_files
)
```

This is dynamic task mapping: each file becomes a mapped transform/embedding task, which scales cleanly as files grow.

## 8) Step-by-Step: Add New Books from Form Fields
The web route `/add` in `main.py` handles manual insertion from form fields (title, year, publisher, description, optional category). The flow first validates duplicates from existing data files, then appends a durable record to `manual_books.txt` so the data survives restarts and can be re-ingested by batch DAGs. After file persistence, the app attempts immediate insertion into Weaviate for instant UI visibility. Finally, it triggers `add_book_data` and `fetch_data` in the background to keep vector index and batch data synchronized. This combined strategy gives fast user feedback plus durable consistency.

Key functions involved:

- `add_book(...)`: route controller and workflow coordinator.
- `load_existing_book_keys()`: duplicate detection source.
- `insert_book_to_weaviate_now(...)`: immediate best-effort write for UX responsiveness.
- `trigger_dag(...)`: non-blocking DAG invocation with API fallback and timeout handling.

## 9) Step-by-Step: Add New Books by Uploading a TXT File
The route `/upload-books` accepts a `.txt` file and validates filename extension and UTF-8 encoding. It parses lines using the expected separator (`:::`), rejects uploads with invalid format, and performs duplicate checks against both current data and repeated records in the same uploaded file. If valid, the file is saved into the data directory, and `fetch_data` is triggered in the background for full transform/embedding/load processing. This workflow is ideal for bulk ingestion because it feeds directly into the same Airflow pipeline used for periodic synchronization.

Key functions and behaviors:

- `parse_books_from_text(...)`: validates line structure and extracts identity fields.
- `load_existing_book_keys()`: prevents duplicate titles across source files.
- `trigger_dag("fetch_data", {})`: initiates asynchronous ingest after successful upload.

## 10) Step-by-Step: Convert Text to Embeddings and Store in Weaviate
Embedding generation is done with `fastembed.TextEmbedding` using model `BAAI/bge-small-en-v1.5`. In batch mode (`fetch_data`), vectors are generated primarily from descriptions, then inserted with metadata into Weaviate as `DataObject`s. In single-book DAG mode (`add_book_data`), vector text includes title, year, publisher, category, and description to represent combined context. Weaviate stores these vectors and properties, enabling hybrid retrieval and semantic ranking. The design also includes fallback behavior: if embeddings are unavailable, search can continue via BM25 keyword retrieval and lexical matching.

Key code idea from `add_book_data.py`:

```python
vector_text = f"{title}. {year}. {publisher}. {final_category}. {description}"
vector = list(embedding_model.embed([vector_text]))[0].tolist()

collection.data.insert(
    properties={
        "title": title,
        "year": year,
        "publisher": publisher,
        "description": description,
        "category": final_category,
    },
    vector=vector,
)
```

## 11) Step-by-Step: Return Recommendations from User Queries
The search route `/search` combines structured filters and semantic relevance. It builds a query from provided fields and attempts Weaviate hybrid search (`BM25 + vector`) when embeddings are available. If embedding generation fails, it falls back to BM25 retrieval; for context-only searches it can evaluate description similarity across local/library data with lexical synonym expansion. After initial retrieval, structured constraints are enforced as true filters to avoid false positives. The route also triggers `query_data` DAG in the background for retrieval observability and logging.

Important functions in `main.py`:

- `search_book(...)`: orchestrates retrieval and filter logic.
- `semantic_filter_by_description(...)`: cosine similarity re-ranking for context.
- `filter_books_locally(...)`: deterministic filters and lexical fallback.
- `context_matches(...)`, `tokenize_for_search(...)`, `expand_context_token(...)`: robust non-vector matching when needed.

## 12) Step-by-Step: Edit Existing Books
The route `/edit-book` updates records by finding the original object with the old identity key `(old_title, old_year, old_publisher)` and then writing new fields. It attempts to regenerate and update vector embedding from the new description; if embedding fails, it still updates metadata to keep editing functional. After Weaviate update, the app updates all `.txt` source files so future re-ingestion does not overwrite edits. This two-layer update strategy (database + source files) is essential for long-term consistency.

Key helper and route:

- `edit_book(...)`: server route for update transaction.
- `update_data_files_after_edit(...)`: rewrites matching lines in source files.

## 13) Step-by-Step: Delete Books
The route `/delete-book` removes a record from Weaviate using a composite filter on title, year, and publisher, then removes matching lines from all source data files. Deleting in both locations avoids reappearance on next synchronization run and keeps the data lake and vector index aligned. Errors during Weaviate deletion are returned as user-visible messages, while file-level IO failures are safely skipped per file to keep operation robust.

Key helper and route:

- `delete_book(...)`: delete request processing and response.
- `update_data_files_after_delete(...)`: source-file cleanup to prevent re-ingest.

## 14) Step-by-Step: Chatbot for Questions and Recommendations
The chatbot endpoint (`/chat` and `/chat-api`) uses retrieval-first generation. It queries Weaviate for top relevant books (hybrid search when possible, BM25 otherwise), formats retrieved items into grounded context, and sends this prompt to Ollama (`llama3.2` by default). If Ollama is unavailable, the app falls back to deterministic recommendation text based on the best retrieved book. This guarantees graceful degradation: users still receive a useful answer even if LLM generation fails.

Key function:

- `generate_chat_answer(message)`: retrieval, prompt assembly, LLM call, and fallback handling.

## 15) Logging and Error Handling Strategy
This project handles errors in layered, practical ways. In Airflow, tasks define retries and retry delays, and task logs are written under `Airflow/logs/dag_id=.../run_id=.../task_id=...`, making root-cause tracing straightforward. In the web app, route-level try/except blocks catch network, connection, and embedding failures to keep UI operations responsive; many non-critical operations (like DAG trigger failures) are treated as best-effort background actions so users can continue working. The code includes schema compatibility checks and defensive collection recreation in DAGs to recover from drift or stale environments. Duplicate prevention is done before writes, reducing downstream cleanup work.

Operational commands you will often use:

```powershell
# container health and status
cd "...\Book-Recommendations-with-RAG-Airflow\Airflow"
docker compose ps

# follow service logs (examples)
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-worker
docker compose logs -f book-ui
docker compose logs -f weaviate

# stop stack
docker compose down
```

## 16) Key Code Blocks to Study First
If you are teaching or onboarding a new engineer, start with these code areas in order:

1. `fetch_data.py`: DAG orchestration, mapped tasks, schema checks, dedupe inserts.
2. `main.py` routes `/upload-books`, `/add`, `/search`, `/edit-book`, `/delete-book`, `/chat-api`.
3. `add_book_data.py`: single-book ingestion and vector creation.
4. `query_data.py`: retrieval DAG and task output logging.
5. `docker-compose.yaml`: service wiring, environment variables, mounted volumes.

This order helps learners understand architecture first, then user workflows, then operational platform concerns.

## 17) Resources
- Apache Airflow docs: https://airflow.apache.org/docs/
- Airflow Docker Compose guide: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
- Weaviate docs: https://weaviate.io/developers/weaviate
- Weaviate Python client docs: https://weaviate.io/developers/weaviate/client-libraries/python
- FastAPI docs: https://fastapi.tiangolo.com/
- FastEmbed docs: https://qdrant.github.io/fastembed/
- Ollama docs: https://github.com/ollama/ollama


