# Weaviate + Airflow Integration Guide

This document explains how to install Weaviate and connect it to your Airflow pipeline using Docker and Python.


## 1. Install Weaviate (Docker Recommended)

### A. Using Docker Compose
- Add the following to your existing `docker-compose.yaml`:

```
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8081:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=20
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=text2vec-transformers
      - TRANSFORMERS_INFERENCE_API=http://localhost:8081
      - ENABLE_MODULES=text2vec-transformers
```

- Start Weaviate:
```
docker compose up -d weaviate
```

### B. Standalone Docker Command
- Run:
```
docker run -d -p 8081:8080 semitechnologies/weaviate:latest
```

## 2. Install Weaviate Python Client in Airflow

- Add to `requirements.txt`:
```
weaviate-client
```
- Or install directly:
```
pip install weaviate-client
```

---

## 3. Connect Airflow Tasks to Weaviate

- In your Airflow DAG Python files, use:

```
import weaviate
client = weaviate.Client("http://localhost:8081")
# Use client to create collections, insert/query data, etc.
```
## 4. Use Weaviate in Your Pipeline

- Add Airflow tasks that interact with Weaviate:

```
from airflow.decorators import task

@task
def insert_to_weaviate():
    import weaviate
    client = weaviate.Client("http://localhost:8081")
    # Insert/query logic here
```

## 5. Restart Airflow if Needed

- If you add new dependencies or change `docker-compose.yaml`, restart Airflow:
```
docker compose down
docker compose up
```
