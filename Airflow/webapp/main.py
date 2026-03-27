from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import math
import os
import re
import requests
from datetime import datetime
import weaviate
from fastembed import TextEmbedding


app = FastAPI(title="Book Recommendation App")
templates = Jinja2Templates(directory="templates")

COLLECTION_NAME = "Books"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embedding_model: TextEmbedding | None = None
embedding_disabled = os.getenv("DISABLE_EMBEDDINGS", "false").strip().lower() == "true"

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

AIRFLOW_API = os.getenv("AIRFLOW_API_URL", "http://airflow-api-server:8080/api/v1")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
BOOK_DATA_DIR = os.getenv("BOOK_DATA_DIR", "/opt/airflow/include/data")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
BOOKS_PER_PAGE = 5


# Lightweight keyword buckets used when category metadata is missing from files
# or user input.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Self-Help": ["mind", "mindful", "habit", "motivation", "self", "growth"],
    "Philosophy": ["philosophy", "meaning", "exist", "stoic", "wisdom"],
    "Science": ["science", "physics", "biology", "technology", "research"],
    "History": ["history", "ancient", "war", "civilization", "century"],
    "Business": ["business", "startup", "marketing", "finance", "leadership"],
    "Fiction": ["novel", "story", "fiction", "fantasy", "mystery"],
}


# Context synonyms support description matching even when the user types a
# related term instead of an exact word from the book description.
CONTEXT_SYNONYMS: dict[str, list[str]] = {
    "startup": ["business", "company", "entrepreneur", "entrepreneurship", "founder"],
    "business": ["startup", "company", "entrepreneur", "finance", "leadership"],
    "entrepreneur": ["startup", "business", "founder", "company"],
    "leadership": ["leader", "management", "business"],
    "finance": ["money", "investing", "investment", "wealth", "business"],
    "marketing": ["brand", "sales", "business", "market"],
    "motivation": ["motivated", "inspiration", "inspire", "drive", "growth"],
    "habit": ["routine", "discipline", "practice", "productivity"],
    "mindset": ["mind", "attitude", "perspective", "thinking"],
    "growth": ["improvement", "progress", "development", "motivation"],
    "science": ["research", "technology", "physics", "biology"],
    "technology": ["tech", "innovation", "science", "digital"],
    "history": ["historical", "ancient", "civilization", "century"],
    "philosophy": ["wisdom", "stoic", "meaning", "ethics"],
    "fiction": ["novel", "story", "fantasy", "mystery"],
}


def normalize_search_token(value: str) -> str:
    # Normalize user text and description text into simple comparable tokens.
    token = re.sub(r"[^a-z0-9]+", "", value.lower())
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def tokenize_for_search(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        token = normalize_search_token(raw_token)
        if len(token) >= 2:
            tokens.append(token)
    return tokens


def expand_context_token(token: str) -> set[str]:
    expanded = {token}
    for related in CONTEXT_SYNONYMS.get(token, []):
        normalized = normalize_search_token(related)
        if normalized:
            expanded.add(normalized)
    return expanded


def context_matches(book: dict, context: str) -> bool:
    # Lexical fallback for topic/context search when semantic embeddings are
    # unavailable or return no usable results.
    context_tokens = tokenize_for_search(context)
    if not context_tokens:
        return True

    # Topic/context search should match description only.
    searchable_text = str(book.get("description", "")).lower()
    searchable_tokens = set(tokenize_for_search(searchable_text))

    # Loose topic matching: one matching context token is enough.
    matched_any = False
    for token in context_tokens:
        related_tokens = expand_context_token(token)
        if any(related in searchable_tokens for related in related_tokens):
            matched_any = True
            continue
        if any(related in searchable_text for related in related_tokens):
            matched_any = True
            continue
        if any(any(related in candidate or candidate in related for candidate in searchable_tokens) for related in related_tokens):
            matched_any = True
            continue

    return matched_any


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_filter_by_description(
    books: list[dict],
    context: str,
    min_similarity: float = 0.12,
) -> list[dict]:
    # Semantic topic matching is restricted to description text only.
    if not context.strip() or not books:
        return books

    descriptions = [str(book.get("description", "")).strip() for book in books]
    valid_indices = [idx for idx, desc in enumerate(descriptions) if desc]
    if not valid_indices:
        return []

    model = get_embedding_model()
    query_vector = list(model.embed([context]))[0].tolist()
    desc_vectors = [
        vector.tolist()
        for vector in model.embed([descriptions[idx] for idx in valid_indices])
    ]

    scored: list[tuple[float, dict]] = []
    for idx, desc_vector in zip(valid_indices, desc_vectors):
        similarity = cosine_similarity(query_vector, desc_vector)
        if similarity >= min_similarity:
            scored.append((similarity, books[idx]))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [book for _, book in scored]


def parse_books_from_text(raw_text: str) -> list[tuple[str, str, str]]:
    books: list[tuple[str, str, str]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(":::")]
        if len(parts) < 6:
            continue
        title, year, publisher = parts[1], parts[2], parts[3]

        if title and year and publisher:
            books.append((title, year, publisher))
    return books


def parse_book_records_from_text(raw_text: str) -> list[dict]:
    # Parse one or more raw text lines into structured book dictionaries used
    # throughout the web app and Airflow-driven fallback paths.
    records: list[dict] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(":::")]
        if len(parts) < 6:
            continue
        title, year, publisher = parts[1], parts[2], parts[3]
        category = normalize_category(parts[4], title, parts[5])
        description = parts[5]
        if not (title and year and publisher):
            continue
        records.append(
            {
                "title": title,
                "year": year,
                "publisher": publisher,
                "category": category,
                "description": description,
            }
        )
    return records


def load_books_from_data_files() -> list[dict]:
    # The UI always merges local .txt books so manual uploads/additions remain
    # visible even when Weaviate or Airflow ingestion is delayed.
    books: list[dict] = []
    if not os.path.isdir(BOOK_DATA_DIR):
        return books

    for name in os.listdir(BOOK_DATA_DIR):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(BOOK_DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                books.extend(parse_book_records_from_text(file_obj.read()))
        except OSError:
            continue

    return dedupe_books(books)


def normalize_book_key(title: str, year: str, publisher: str) -> tuple[str, str, str]:
    return (title.strip().lower(), year.strip().lower(), publisher.strip().lower())


def load_existing_book_keys() -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not os.path.isdir(BOOK_DATA_DIR):
        return keys

    for name in os.listdir(BOOK_DATA_DIR):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(BOOK_DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                file_books = parse_books_from_text(file_obj.read())
            for title, year, publisher in file_books:
                keys.add(normalize_book_key(title, year, publisher))
        except OSError:
            continue
    return keys


def dedupe_books(books: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for book in books:
        title = str(book.get("title", "")).strip()
        year = str(book.get("year", "")).strip()
        publisher = str(book.get("publisher", "")).strip()
        key = normalize_book_key(title, year, publisher)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(book)
    return deduped


def trigger_dag(dag_id: str, conf: dict) -> tuple[int, str]:
    """Trigger a DAG in Airflow. Non-blocking with error handling."""
    # Try both API versions because the active Airflow image and reverse proxy
    # path may expose either /api/v1 or /api/v2.
    base_url = AIRFLOW_API.rstrip("/")
    urls = [f"{base_url}/dags/{dag_id}/dagRuns"]
    if "/api/v1" in base_url:
        urls.append(f"{base_url.replace('/api/v1', '/api/v2')}/dags/{dag_id}/dagRuns")
    elif "/api/v2" in base_url:
        urls.append(f"{base_url.replace('/api/v2', '/api/v1')}/dags/{dag_id}/dagRuns")

    payload = {"conf": conf}
    last_status = 500
    last_error = "Unknown Airflow API error"
    for url in urls:
        try:
            response = requests.post(
                url,
                json=payload,
                auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
                timeout=5,
                verify=False,
            )
            if 200 <= response.status_code < 400:
                return response.status_code, "DAG triggered"
            last_status = response.status_code
            last_error = f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            last_status = 408
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_status = 503
            last_error = "Connection error"
        except Exception as exc:  # noqa: BLE001
            last_status = 500
            last_error = str(exc)

    return last_status, last_error


def connect_client():
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
        skip_init_checks=True,
    )


def get_embedding_model() -> TextEmbedding:
    global embedding_model
    global embedding_disabled

    if embedding_disabled:
        raise RuntimeError("Embeddings are temporarily disabled; using BM25 fallback.")

    if embedding_model is None:
        try:
            embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
        except Exception:
            embedding_disabled = True
            raise
    return embedding_model


def fetch_all_books() -> list[dict]:
    # Primary read path for the UI: fetch from Weaviate, then merge local file
    # data so the interface remains usable during background sync delays.
    books: list[dict] = []
    try:
        with connect_client() as client:
            collection = client.collections.get(COLLECTION_NAME)
            results = collection.query.fetch_objects(
                limit=10000,
                return_properties=["title", "year", "publisher", "category", "description"],
            )
        books = [obj.properties for obj in results.objects] if results.objects else []
    except Exception:
        books = []

    # Always merge books from local txt files so manual_books.txt is visible
    # even when Airflow/Weaviate ingestion is delayed.
    books.extend(load_books_from_data_files())

    for book in books:
        book["category"] = normalize_category(
            str(book.get("category", "")),
            str(book.get("title", "")),
            str(book.get("description", "")),
        )
    return dedupe_books(books)


def filter_books_locally(
    books: list[dict],
    title: str,
    year: str,
    publisher: str,
    category: str,
    context: str,
) -> list[dict]:
    title_q = title.lower().strip()
    year_q = year.lower().strip()
    publisher_q = publisher.lower().strip()
    category_q = category.lower().strip()
    context_q = context.lower().strip()

    filtered: list[dict] = []
    for book in books:
        b_title = str(book.get("title", "")).strip().lower()
        b_year = str(book.get("year", "")).lower()
        b_publisher = str(book.get("publisher", "")).lower()
        b_category = str(book.get("category", "")).lower()
        b_description = str(book.get("description", "")).lower()

        if title_q and title_q != b_title:
            continue
        if year_q and year_q not in b_year:
            continue
        if publisher_q and publisher_q not in b_publisher:
            continue
        if category_q and category_q not in b_category:
            continue
        if context_q and not context_matches(book, context_q):
            continue

        filtered.append(book)

    return dedupe_books(filtered)


def paginate_books(books: list[dict], page: int) -> tuple[list[dict], int, int]:
    total_books = len(books)
    total_pages = max(1, (total_books + BOOKS_PER_PAGE - 1) // BOOKS_PER_PAGE)
    current_page = min(max(page, 1), total_pages)
    start = (current_page - 1) * BOOKS_PER_PAGE
    end = start + BOOKS_PER_PAGE
    return books[start:end], current_page, total_pages


def detect_categories(book: dict) -> list[str]:
    text = (
        f"{book.get('title', '')} {book.get('description', '')}"
    ).strip().lower()
    matched: list[str] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched.append(category)
    return matched or ["General"]


def normalize_category(category: str, title: str = "", description: str = "") -> str:
    clean = "/".join(
        [part.strip() for part in category.split("/") if part.strip()]
    )
    if clean:
        return clean
    return "/".join(detect_categories({"title": title, "description": description}))


def build_library_stats(books: list[dict]) -> dict:
    unique_publishers: set[str] = set()
    category_counts: dict[str, int] = {}

    for book in books:
        publisher = str(book.get("publisher", "")).strip()
        if publisher:
            unique_publishers.add(publisher)
        category_value = normalize_category(
            str(book.get("category", "")),
            str(book.get("title", "")),
            str(book.get("description", "")),
        )
        for category in [part.strip() for part in category_value.split("/") if part.strip()]:
            category_counts[category] = category_counts.get(category, 0) + 1

    top_categories = sorted(
        category_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "total_books": len(books),
        "total_publishers": len(unique_publishers),
        "categories": top_categories,
    }


def render_index(request: Request, page: int = 1, **context):
    # Central page renderer used by most routes so library stats, management
    # lists, and per-route messages stay consistent.
    all_books = fetch_all_books()
    manage_books, current_page, total_pages = paginate_books(all_books, page)
    stats = build_library_stats(all_books)
    base_context = {
        "request": request,
        "all_books": all_books,
        "manage_books": manage_books,
        "current_page": current_page,
        "total_pages": total_pages,
        "stats": stats,
    }
    base_context.update(context)
    return templates.TemplateResponse("index.html", base_context)


def insert_book_to_weaviate_now(
    title: str,
    year: str,
    publisher: str,
    description: str,
    category: str,
) -> tuple[bool, str]:
    """Best-effort immediate insert so users can see newly added books right away."""
    try:
        from weaviate.classes.query import Filter

        with connect_client() as client:
            collection = client.collections.get(COLLECTION_NAME)
            where_filter = (
                Filter.by_property("title").equal(title)
                & Filter.by_property("year").equal(year)
                & Filter.by_property("publisher").equal(publisher)
            )
            existing = collection.query.fetch_objects(
                filters=where_filter,
                limit=1,
                return_properties=["title"],
            )
            if existing.objects:
                return True, ""

            collection.data.insert(
                properties={
                    "title": title,
                    "year": year,
                    "publisher": publisher,
                    "description": description,
                    "category": category,
                }
            )
        return True, ""
    except Exception as exc:
        return False, str(exc)


def generate_chat_answer(message: str) -> tuple[str, list[dict]]:
    # Chat retrieval mirrors search behavior: prefer hybrid/vector retrieval,
    # then fall back to BM25 so the assistant still works without embeddings.
    import ollama as ollama_client

    with connect_client() as client:
        collection = client.collections.get(COLLECTION_NAME)
        try:
            from weaviate.classes.query import HybridFusion

            vector = list(get_embedding_model().embed([message]))[0].tolist()
            results = collection.query.hybrid(
                query=message,
                vector=vector,
                limit=3,
                alpha=0.5,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_properties=["title", "year", "publisher", "category", "description"],
            )
        except Exception:
            results = collection.query.bm25(
                query=message,
                limit=3,
                return_properties=["title", "year", "publisher", "category", "description"],
            )

    retrieved_books = [obj.properties for obj in results.objects] if results.objects else []
    for book in retrieved_books:
        book["category"] = normalize_category(
            str(book.get("category", "")),
            str(book.get("title", "")),
            str(book.get("description", "")),
        )

    if not retrieved_books:
        try:
            trigger_dag("query_data", {"query": message})
        except Exception:
            pass
        return "I could not find any matching books in the library right now.", []

    context_lines = []
    for i, b in enumerate(retrieved_books, 1):
        context_lines.append(
            f"{i}. Title: {b.get('title', '')}\n"
            f"   Year: {b.get('year', '')}\n"
            f"   Publisher: {b.get('publisher', '')}\n"
            f"   Category: {b.get('category', '')}\n"
            f"   Description: {b.get('description', '')}"
        )
    context_text = "\n\n".join(context_lines)

    prompt = (
        f"You are a helpful book recommendation assistant.\n"
        f"A user asked: \"{message}\"\n\n"
        f"Based on the following books from our library, write a friendly, "
        f"personalised recommendation in 2-3 sentences. "
        f"Mention the best matching book by title, year, publisher, and category, and explain why it suits the user's request.\n\n"
        f"Available books:\n{context_text}\n\n"
        f"Recommendation:"
    )

    try:
        oc = ollama_client.Client(host=OLLAMA_HOST)
        response = oc.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = response["message"]["content"].strip()
    except Exception:
        best = retrieved_books[0]
        reply = (
            f"I recommend '{best.get('title')}' ({best.get('year')}, {best.get('publisher')}, {best.get('category')}). "
            f"{best.get('description')}"
        )

    try:
        trigger_dag("query_data", {"query": message})
    except Exception:
        pass

    return reply, retrieved_books


@app.get("/", response_class=HTMLResponse)
def home(request: Request, page: int = 1):
    """Serve home page with all books pre-loaded."""
    return render_index(request, page=page)


@app.post("/upload-books", response_class=HTMLResponse)
async def upload_books_file(request: Request, books_file: UploadFile = File(...)):
    if not books_file.filename:
        return render_index(request, message="Please select a .txt file to upload.")

    if not books_file.filename.lower().endswith(".txt"):
        return render_index(request, message="Only .txt files are supported.")

    content_bytes = await books_file.read()
    try:
        content_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return render_index(request, message="File must be UTF-8 text.")

    uploaded_books = parse_books_from_text(content_text)
    if not uploaded_books:
        return render_index(
            request,
            message="No valid records found. Expected lines like id:::title:::year:::publisher:::category:::description.",
        )

    existing_keys = load_existing_book_keys()
    seen_in_upload: set[tuple[str, str, str]] = set()
    duplicates: list[tuple[str, str, str]] = []

    for title, year, publisher in uploaded_books:
        key = normalize_book_key(title, year, publisher)
        if key in seen_in_upload or key in existing_keys:
            duplicates.append((title, year, publisher))
        else:
            seen_in_upload.add(key)

    if duplicates:
        duplicate_preview = ", ".join([f"{t} ({y}, {p})" for t, y, p in duplicates[:3]])
        extra = "" if len(duplicates) <= 3 else f" and {len(duplicates) - 3} more"
        return render_index(
            request,
            message=f"Upload rejected. Duplicate title+year+publisher found: {duplicate_preview}{extra}.",
        )

    os.makedirs(BOOK_DATA_DIR, exist_ok=True)
    safe_name = os.path.basename(books_file.filename)
    target_path = os.path.join(BOOK_DATA_DIR, safe_name)
    if os.path.exists(target_path):
        stem, ext = os.path.splitext(safe_name)
        safe_name = f"{stem}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{ext}"
        target_path = os.path.join(BOOK_DATA_DIR, safe_name)

    with open(target_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(content_text)

    # Trigger DAG in background (non-blocking, ignore failures)
    try:
        trigger_dag("fetch_data", {})
    except Exception:
        pass
    return render_index(
        request,
        message="Books file uploaded successfully. The library will be updated shortly.",
    )


MANUAL_BOOKS_FILE = os.path.join(BOOK_DATA_DIR, "manual_books.txt")


@app.post("/add", response_class=HTMLResponse)
def add_book(
    request: Request,
    title: str = Form(...),
    year: str = Form(...),
    publisher: str = Form(...),
    description: str = Form(...),
    category: str = Form(""),
):
    # Add flow writes to a durable text file first, then tries immediate
    # Weaviate insertion so the new book can appear right away in the UI.
    # Reject if already present in any existing data file
    existing_keys = load_existing_book_keys()
    key = normalize_book_key(title, year, publisher)
    if key in existing_keys:
        return render_index(request, message=f"'{title}' ({year}, {publisher}) is already in the library.")

    # Persist to manual_books.txt so fetch_data can load it and
    # it survives Weaviate restarts
    os.makedirs(BOOK_DATA_DIR, exist_ok=True)
    final_category = normalize_category(category, title, description)
    with open(MANUAL_BOOKS_FILE, "a", encoding="utf-8") as f:
        uid = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        f.write(f"{uid}:::{title}:::{year}:::{publisher}:::{final_category}:::{description}\n")

    inserted_now, insert_error = insert_book_to_weaviate_now(
        title=title,
        year=year,
        publisher=publisher,
        description=description,
        category=final_category,
    )

    # Trigger Airflow DAGs in background:
    # 1) add_book_data for quick single-book ingest
    # 2) fetch_data for full pipeline (transform -> embed -> load)
    dag_errors: list[str] = []
    try:
        status_code, dag_message = trigger_dag(
        "add_book_data",
        {
            "title": title,
            "year": year,
            "publisher": publisher,
            "description": description,
            "category": final_category,
        },
        )
        if status_code >= 400:
            dag_errors.append(f"add_book_data -> {dag_message}")
    except Exception as exc:
        dag_errors.append(f"add_book_data trigger exception -> {exc}")

    try:
        status_code, dag_message = trigger_dag("fetch_data", {})
        if status_code >= 400:
            dag_errors.append(f"fetch_data -> {dag_message}")
    except Exception as exc:
        dag_errors.append(f"fetch_data trigger exception -> {exc}")

    if dag_errors:
        print(f"Background DAG trigger issues after add_book: {'; '.join(dag_errors)}")

    if inserted_now:
        return render_index(
            request,
            message="Book added successfully and is visible now. Background sync started.",
        )

    return render_index(
        request,
        message=(
            "Book saved to file, but immediate display failed. "
            "It should appear after background sync. "
            f"Immediate insert issue: {insert_error}."
        ),
    )


@app.post("/search", response_class=HTMLResponse)
def search_book(
    request: Request,
    title: str = Form(""),
    year: str = Form(""),
    publisher: str = Form(""),
    category: str = Form(""),
    context: str = Form(""),
):
    # Search combines structured field filters with description-only topic
    # matching, using semantic similarity first and lexical fallback second.
    title = title.strip()
    year = year.strip()
    publisher = publisher.strip()
    raw_category = category.strip()
    category = normalize_category(raw_category) if raw_category else ""
    context = context.strip()

    parts = [p for p in [title, year, publisher, category, context] if p]
    if not parts:
        return render_index(request, message="Please fill in at least one search field.")

    # Build keyword query: prefer exact name/title fields when no free-text context
    keyword_parts = [p for p in [title, year, publisher, category] if p]
    keyword_query = " ".join(keyword_parts) if keyword_parts else context
    structured_filters = any([title, year, publisher, category])

    # Full query for vector embedding combines all fields
    full_query = ". ".join(parts)

    # For context-only search, evaluate against the full library descriptions
    # so topic/context doesn't depend on top-k retrieval from Weaviate.
    if context and not structured_filters:
        books = fetch_all_books()
    else:
        with connect_client() as client:
            collection = client.collections.get(COLLECTION_NAME)
            try:
                from weaviate.classes.query import HybridFusion

                vector = list(get_embedding_model().embed([full_query]))[0].tolist()
                results = collection.query.hybrid(
                    query=keyword_query,
                    vector=vector,
                    limit=5,
                    alpha=0.5,  # 0=pure BM25 keyword, 1=pure vector; 0.5=balanced
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    return_properties=["title", "year", "publisher", "category", "description"],
                )
            except Exception:
                # Fallback when embedding model cannot be loaded (e.g. no HF access).
                results = collection.query.bm25(
                    query=keyword_query,
                    limit=5,
                    return_properties=["title", "year", "publisher", "category", "description"],
                )

        books = [obj.properties for obj in results.objects] if results.objects else []
        for book in books:
            book["category"] = normalize_category(
                str(book.get("category", "")),
                str(book.get("title", "")),
                str(book.get("description", "")),
            )
    # Include local-file books in search so manual_books.txt is always searchable.
    local_matches = filter_books_locally(
        books=load_books_from_data_files(),
        title=title,
        year=year,
        publisher=publisher,
        category=category,
        context=context,
    )
    books.extend(local_matches)
    books = dedupe_books(books)

    # Structured fields should behave like actual filters, not just ranking hints.
    if structured_filters:
        books = filter_books_locally(
            books=books,
            title=title,
            year=year,
            publisher=publisher,
            category=category,
            context="",
        )

    # Topic/context should search by description semantic similarity when possible.
    if context:
        try:
            semantic_books = semantic_filter_by_description(books, context)
            if semantic_books:
                books = semantic_books
            else:
                books = filter_books_locally(
                    books=books,
                    title="",
                    year="",
                    publisher="",
                    category="",
                    context=context,
                )
        except Exception:
            # Fallback to lexical description matching when embeddings are unavailable.
            books = filter_books_locally(
                books=books,
                title="",
                year="",
                publisher="",
                category="",
                context=context,
            )

    # Trigger logging DAG (non-blocking, ignore failures)
    try:
        trigger_dag("query_data", {"query": full_query})
    except Exception:
        pass

    return render_index(
        request,
        books=books,
        search_title=title,
        search_year=year,
        search_publisher=publisher,
        search_category=raw_category,
        search_context=context,
    )


@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, message: str = Form(...)):
    reply, retrieved_books = generate_chat_answer(message)
    return render_index(request, chat_reply=reply, chat_books=retrieved_books)


@app.post("/chat-api")
def chat_api(message: str = Form(...)):
    message = message.strip()
    if not message:
        return JSONResponse(
            status_code=400,
            content={"reply": "Please enter a question.", "books": []},
        )

    reply, retrieved_books = generate_chat_answer(message)
    source_books = [
        {
            "title": str(b.get("title", "")).strip(),
            "year": str(b.get("year", "")).strip(),
            "publisher": str(b.get("publisher", "")).strip(),
            "category": str(b.get("category", "")).strip(),
        }
        for b in retrieved_books
    ]
    return JSONResponse(content={"reply": reply, "books": source_books})


@app.get("/list-books", response_class=HTMLResponse)
def list_books(request: Request, page: int = 1):
    """Fetch all books from Weaviate for management."""
    return render_index(request, page=page)


def update_data_files_after_delete(title: str, year: str, publisher: str) -> None:
    """Remove book from all data files by title+year+publisher."""
    if not os.path.isdir(BOOK_DATA_DIR):
        return

    for name in os.listdir(BOOK_DATA_DIR):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(BOOK_DATA_DIR, name)
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = [p.strip() for p in line.split(":::")]
                if len(parts) >= 4:
                    line_title = parts[1]
                    line_year = parts[2]
                    line_publisher = parts[3]
                    if (
                        line_title.lower() == title.lower()
                        and line_year.lower() == year.lower()
                        and line_publisher.lower() == publisher.lower()
                    ):
                        continue  # Skip this line (delete it)
                new_lines.append(line)

            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except OSError:
            continue


def update_data_files_after_edit(
    old_title: str,
    old_year: str,
    old_publisher: str,
    new_title: str,
    new_year: str,
    new_publisher: str,
    new_description: str,
    new_category: str,
) -> None:
    """Update book in all data files by old title+year+publisher."""
    if not os.path.isdir(BOOK_DATA_DIR):
        return

    for name in os.listdir(BOOK_DATA_DIR):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(BOOK_DATA_DIR, name)
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = [p.strip() for p in line.split(":::")]
                if len(parts) >= 6:
                    uid = parts[0]
                    line_title = parts[1]
                    line_year = parts[2]
                    line_publisher = parts[3]
                    if (
                        line_title.lower() == old_title.lower()
                        and line_year.lower() == old_year.lower()
                        and line_publisher.lower() == old_publisher.lower()
                    ):
                        # Update this line
                        new_line = (
                            f"{uid}:::{new_title}:::{new_year}:::{new_publisher}:::{new_category}:::{new_description}\n"
                        )
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except OSError:
            continue


@app.post("/delete-book", response_class=HTMLResponse)
def delete_book(
    request: Request,
    title: str = Form(...),
    year: str = Form(...),
    publisher: str = Form(...),
    page: int = Form(1),
):
    """Delete a book from Weaviate and all data files."""
    try:
        with connect_client() as client:
            from weaviate.classes.query import Filter

            collection = client.collections.get(COLLECTION_NAME)
            where_filter = (
                Filter.by_property("title").equal(title)
                & Filter.by_property("year").equal(year)
                & Filter.by_property("publisher").equal(publisher)
            )
            collection.data.delete_many(
                where=where_filter
            )
    except Exception as exc:
        return render_index(request, page=page, message=f"Error deleting book: {exc}")

    # Remove from data files
    update_data_files_after_delete(title, year, publisher)

    return render_index(
        request,
        page=page,
        message=f"Book '{title}' ({year}, {publisher}) has been deleted successfully.",
    )


@app.post("/edit-book", response_class=HTMLResponse)
def edit_book(
    request: Request,
    old_title: str = Form(...),
    old_year: str = Form(...),
    old_publisher: str = Form(...),
    new_title: str = Form(...),
    new_year: str = Form(...),
    new_publisher: str = Form(...),
    new_description: str = Form(...),
    new_category: str = Form(""),
    page: int = Form(1),
):
    """Edit a book in Weaviate and all data files."""
    # Edit flow updates Weaviate first, then mirrors the change back into the
    # source text files so future re-ingestion preserves the edit.
    try:
        with connect_client() as client:
            from weaviate.classes.query import Filter

            collection = client.collections.get(COLLECTION_NAME)
            # Find the book by old title+year+publisher
            where_filter = (
                Filter.by_property("title").equal(old_title)
                & Filter.by_property("year").equal(old_year)
                & Filter.by_property("publisher").equal(old_publisher)
            )
            results = collection.query.fetch_objects(
                filters=where_filter,
                limit=1,
            )

            if results.objects:
                obj_id = results.objects[0].uuid
                final_category = normalize_category(new_category, new_title, new_description)
                update_properties = {
                    "title": new_title,
                    "year": new_year,
                    "publisher": new_publisher,
                    "description": new_description,
                    "category": final_category,
                }

                try:
                    # Try updating vector when embeddings are available.
                    new_embedding = list(
                        get_embedding_model().embed([new_description])
                    )[0].tolist()
                    collection.data.update(
                        uuid=obj_id,
                        properties=update_properties,
                        vector=new_embedding,
                    )
                except Exception:
                    # Fallback: keep edit functional without embeddings.
                    collection.data.update(
                        uuid=obj_id,
                        properties=update_properties,
                    )
    except Exception as exc:
        return render_index(request, page=page, message=f"Error updating book: {exc}")

    # Update data files
    update_data_files_after_edit(
        old_title,
        old_year,
        old_publisher,
        new_title,
        new_year,
        new_publisher,
        new_description,
        normalize_category(new_category, new_title, new_description),
    )

    return render_index(request, page=page, message="Book updated successfully.")
