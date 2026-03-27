# Apache Airflow Docker Compose Setup Guide

This guide will help you set up Apache Airflow using Docker Compose, which is the easiest and most reliable way to run Airflow locally.

---

## 1. Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux).
- Install [Docker Compose](https://docs.docker.com/compose/) (if not included with Docker Desktop).

---

## 2. Download the Official Airflow Docker Compose Files

run in bash

```
curl.exe -LfO https://airflow.apache.org/docs/apache-airflow/2.8.3/docker-compose.yaml
```

or run in a powershell

```
Invoke-WebRequest https://airflow.apache.org/docs/apache-airflow/2.8.3/docker-compose.yaml -OutFile docker-compose.yaml
```

Or download from: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

---

## 3. Set Up the Airflow Environment

Create a folder for Airflow and move the `docker-compose.yaml` file there. Then, in that folder, run in bash:

```
mkdir -p ./dags ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

On Windows, you can manually create the folders and set the environment variable in the `.env` file:

```
AIRFLOW_UID=50000
```

---

## 4. Initialize the Airflow Database

```
docker compose up airflow-init
```

---

## 5. Start Airflow

```
docker compose up
```

---

## 6. Access the Airflow UI
- Open your browser and go to: http://localhost:8080
- Default username: `airflow`
- Default password: `airflow`

---

## 7. Add Your DAGs
- Place your DAG Python files (e.g., `my_first_dag.py`, `my_second_dag.py`) in the `dags` folder you created.
- The Airflow UI will automatically detect and show your DAGs.

---

## 8. Stop Airflow
- Press `Ctrl+C` in the terminal running Docker Compose.
- To remove containers and networks:
  ```
  docker compose down
  ```

---

For more details, see the [official Airflow Docker documentation](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html).

---------
# Airflow Custom Docker Image Guide

If you need extra Python libraries in Airflow, use a **custom Docker image**.

## Why a custom image
The default config uses:

- `image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:2.8.3}`

That pulls a prebuilt image and does not include your project-specific packages unless installed at runtime.

## Recommended approach

1. In the `Airflow` folder, create:
   - `Dockerfile`
   - `requirements.txt`

2. Add required libraries to `requirements.txt`.

3. In `Airflow/docker-compose.yaml`, enable custom build:
   - Keep `build: .`
   - Comment/remove `image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:2.8.3}`

This makes all Airflow services use your customized image.

## Example files

### `Airflow/requirements.txt`
```txt
weaviate-client
pandas
numpy
```

### `Airflow/Dockerfile`
```dockerfile
FROM apache/airflow:2.8.3

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
```

### `Airflow/docker-compose.yaml` (`x-airflow-common`)
```yaml
x-airflow-common:
  &airflow-common
  # image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:2.8.3}
  build: .
```

## Rebuild and run (Windows / VS Code terminal)

```powershell
cd "d:\1-My Goals\8-Funding for My Desire of My Main-Job and Second-Job-Entrepreneur\4-Git_Projects_Mypage\Gitcoding\Airflow-RAG for Book Recommendations\Airflow"
docker compose build --no-cache
docker compose up -d
```

## Verify package installation

```powershell
docker compose run --rm airflow-webserver python -c "import weaviate; print('weaviate import OK')"
```

## Notes
- Prefer custom image over `_PIP_ADDITIONAL_REQUIREMENTS` for stable setups.
- Rebuild the image whenever `requirements.txt` changes.
- Pin package versions in `requirements.txt` for reproducibility.