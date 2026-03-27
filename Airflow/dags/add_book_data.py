from airflow.sdk import dag, task
from airflow.operators.python import get_current_context
from datetime import datetime
import weaviate
from fastembed import TextEmbedding


COLLECTION_NAME = "Books"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# Keyword fallback used to auto-fill category when the incoming DAG payload
# does not provide one explicitly.
CATEGORY_KEYWORDS = {
    "Self-Help": ["mind", "mindful", "habit", "motivation", "self", "growth"],
    "Philosophy": ["philosophy", "meaning", "exist", "stoic", "wisdom"],
    "Science": ["science", "physics", "biology", "technology", "research"],
    "History": ["history", "ancient", "war", "civilization", "century"],
    "Business": ["business", "startup", "marketing", "finance", "leadership"],
    "Fiction": ["novel", "story", "fiction", "fantasy", "mystery"],
}


def infer_category(title: str, description: str) -> str:
    # Infer one or more categories from title/description text when the UI did
    # not pass a category in dag_run.conf.
    text = f"{title} {description}".strip().lower()
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched.append(category)
    return "/".join(matched) if matched else "General"


def normalize_category(category: str, title: str, description: str) -> str:
    # Preserve user-provided categories if present; otherwise infer them.
    clean = "/".join([part.strip() for part in category.split("/") if part.strip()])
    return clean or infer_category(title, description)


@dag(
    start_date=datetime(2026, 3, 24),
    schedule=None,
    catchup=False,
    tags=["books", "ingestion"],
)
def add_book_data():

    @task
    def add_book():
        # Airflow passes UI form data through dag_run.conf when this DAG is
        # triggered from the FastAPI app.
        context = get_current_context()
        conf = context.get("dag_run").conf or {}

        title = conf.get("title", "").strip()
        year = conf.get("year", "").strip()
        publisher = conf.get("publisher", "").strip()
        description = conf.get("description", "").strip()
        category = conf.get("category", "").strip()

        if not title or not year or not publisher or not description:
            raise ValueError("title, year, publisher, and description are required in dag_run.conf")

        final_category = normalize_category(category, title, description)

        # This DAG performs a direct single-book insert into Weaviate so the
        # newly added book can appear before the full fetch_data pipeline runs.
        with weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
            skip_init_checks=True,
        ) as client:
            embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
            collection = client.collections.get(COLLECTION_NAME)

            # Keep the vector text aligned with the fields users search on.
            vector_text = f"{title}. {year}. {publisher}. {final_category}. {description}"
            vector = list(embedding_model.embed([vector_text]))[0].tolist()

            uid = collection.data.insert(
                properties={
                    "title": title,
                    "year": year,
                    "publisher": publisher,
                    "description": description,
                    "category": final_category,
                },
                vector=vector,
            )

        # Returning a small payload makes task output easier to inspect in the
        # Airflow UI and logs.
        return {"id": str(uid), "title": title}

    add_book()


add_book_data()
