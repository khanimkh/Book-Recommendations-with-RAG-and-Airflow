from airflow.sdk import dag, task, Asset, chain
from weaviate.classes.data import DataObject
from weaviate.classes.config import Property, DataType
from pendulum import datetime, duration

import os
import weaviate

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/opt/airflow/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def create_books_collection_with_schema(client) -> None:
    # Centralized schema creation keeps both the setup task and the load task
    # aligned on the same expected properties.
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.TEXT),
            Property(name="publisher", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
        ],
    )


# Keyword-based category hints are used whenever the source file does not
# provide a category or provides an empty one.
CATEGORY_KEYWORDS = {
    "Self-Help": ["mind", "mindful", "habit", "motivation", "self", "growth"],
    "Philosophy": ["philosophy", "meaning", "exist", "stoic", "wisdom"],
    "Science": ["science", "physics", "biology", "technology", "research"],
    "History": ["history", "ancient", "war", "civilization", "century"],
    "Business": ["business", "startup", "marketing", "finance", "leadership"],
    "Fiction": ["novel", "story", "fiction", "fantasy", "mystery"],
}


def infer_category(title: str, description: str) -> str:
    # Infer one or more categories from the book text when category metadata is
    # missing from the raw input files.
    text = f"{title} {description}".strip().lower()
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched.append(category)
    return "/".join(matched) if matched else "General"


def normalize_category(category: str, title: str, description: str) -> str:
    # Preserve user/file categories if present; otherwise auto-detect them.
    clean = "/".join([part.strip() for part in category.split("/") if part.strip()])
    return clean or infer_category(title, description)


# Hourly DAG that reads every book .txt file, transforms rows into structured
# records, generates description embeddings, and loads them into Weaviate.
@dag(
    start_date=datetime(2026, 3, 18),
    schedule="@hourly",
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10)
    }
)
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        # Ensure the Weaviate collection exists before any mapped tasks run.
        # This task also self-heals stale schemas after resets or code changes.
        #hook = WeaviateHook("my_weaviate_conn")
        #client = hook.get_conn()

        client = weaviate.connect_to_local(
           host="weaviate",
           port=8080,
           grpc_port=50051,
           skip_init_checks=True,)

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        required_props = ["title", "year", "publisher", "category", "description"]

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            create_books_collection_with_schema(client)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            return

        # If an older schema exists (for example title/author/description only),
        # drop and recreate to guarantee the new properties are available.
        try:
            collection = client.collections.get(COLLECTION_NAME)
            collection.query.fetch_objects(limit=1, return_properties=required_props)
            print(f"Collection {COLLECTION_NAME} schema is compatible.")
        except Exception as exc:
            print(f"Schema mismatch detected for {COLLECTION_NAME}: {exc}")
            print("Recreating collection with the current schema...")
            client.collections.delete(COLLECTION_NAME)
            create_books_collection_with_schema(client)
            print(f"Collection {COLLECTION_NAME} recreated successfully.")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        # Dynamic task mapping starts here: each discovered .txt file becomes
        # its own transform and embedding task downstream.
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> str:
        # Parse one raw input file into normalized dictionaries used by later
        # embedding and load tasks.
        import os

        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        rows = []
        for row in book_descriptions:
            parts = [part.strip() for part in row.split(":::")]
            if len(parts) < 6:
                continue
            title = parts[1]
            year = parts[2]
            publisher = parts[3]
            description = parts[5]
            category = normalize_category(parts[4], title, description)
            rows.append(
                {
                    "title": title,
                    "year": year,
                    "publisher": publisher,
                    "description": description,
                    "category": category,
                }
            )

        return rows

    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        # Embeddings are generated from descriptions because topic/context
        # search is driven primarily by the descriptive text.
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)


        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]



        return description_embeddings

    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")]
    )
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        # Load each transformed row and its matching embedding into Weaviate.
        # This task also guards against schema drift and duplicate inserts.
        client = weaviate.connect_to_local(
           host="weaviate",
           port=8080,
           grpc_port=50051,
           skip_init_checks=True,)
        
        required_props = ["title", "year", "publisher", "category", "description"]
        collection = client.collections.get(COLLECTION_NAME)

        # Defensive check: this task can run against a stale class schema
        # after resets/reloads. Recreate class if required properties are missing.
        try:
            collection.query.fetch_objects(limit=1, return_properties=required_props)
        except Exception as exc:
            print(f"Schema mismatch detected in load task for {COLLECTION_NAME}: {exc}")
            print("Recreating collection with the current schema in load task...")
            client.collections.delete(COLLECTION_NAME)
            create_books_collection_with_schema(client)
            collection = client.collections.get(COLLECTION_NAME)

        existing_keys: set[tuple[str, str, str]] = set()
        # Existing title/year/publisher keys are used as a lightweight dedupe
        # check so reruns do not keep inserting the same books.
        try:
            existing_objects = collection.query.fetch_objects(
                limit=10000,
                return_properties=["title", "year", "publisher"],
            )
        except Exception as exc:
            print(f"Existing objects query failed due to schema mismatch: {exc}")
            print("Recreating collection and continuing with fresh insert...")
            client.collections.delete(COLLECTION_NAME)
            create_books_collection_with_schema(client)
            collection = client.collections.get(COLLECTION_NAME)
            existing_keys = set()
            existing_objects = None
        for obj in (existing_objects.objects if existing_objects else []):
            title = str(obj.properties.get("title", "")).strip().lower()
            year = str(obj.properties.get("year", "")).strip().lower()
            publisher = str(obj.properties.get("publisher", "")).strip().lower()
            if title and year and publisher:
                existing_keys.add((title, year, publisher))

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                # The combination of title/year/publisher acts as the logical
                # unique key for one book record in this project.
                key = (
                    str(book_data["title"]).strip().lower(),
                    str(book_data["year"]).strip().lower(),
                    str(book_data["publisher"]).strip().lower(),
                )
                if key in existing_keys:
                    continue

                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "year": book_data["year"],
                        "publisher": book_data["publisher"],
                        "description": book_data["description"],
                        "category": book_data.get("category", "General"),
                    },
                    vector=emb,
                )
                items.append(item)
                existing_keys.add(key)

            if items:
                collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    # Make the collection setup happen before the final load stage. The mapped
    # transform/embed tasks feed data into the load task in between.
    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()