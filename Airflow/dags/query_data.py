from airflow.sdk import dag, task, Asset
from airflow.operators.python import get_current_context
from datetime import datetime
import weaviate 
from fastembed import TextEmbedding


COLLECTION_NAME = "Books"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# This DAG is used as a lightweight retrieval/logging step. It runs whenever
# the book vector asset is updated, and it can also be triggered manually with
# a query string in dag_run.conf.
@dag(
    start_date=datetime(2026, 3, 24),
    schedule=[Asset("my_book_vector_data")],
    catchup=False,
    tags=["books", "search"],
)
def query_data():
    
    @task
    def search_vector_db_for_a_book():
        # Read the user/query payload passed when the DAG is triggered from the
        # web application. Fall back to a default example query for manual runs.
        context = get_current_context()
        query_str = (context.get("dag_run").conf or {}).get("query", "A philosophical book")

        with weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
            skip_init_checks=True,
        ) as client:
            # Build an embedding for the query text and search the Weaviate
            # collection for the nearest stored book vectors.
            embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
            collection = client.collections.get(COLLECTION_NAME)

            query_emb = list(embedding_model.embed([query_str]))[0].tolist()

            results = collection.query.near_vector(
                near_vector=query_emb,
                limit=3,
                return_properties=["title", "year", "publisher", "category", "description"],
            )

            if not results.objects:
                # Returning a consistent empty payload makes this task easier to
                # inspect from Airflow logs or downstream consumers.
                print("No matching book found.")
                return {"query": query_str, "results": []}

            formatted_results = [obj.properties for obj in results.objects]
            top_result = formatted_results[0]
            # Log the top result for quick visibility in the Airflow task log.
            print(
                f"You should read: {top_result['title']} "
                f"({top_result.get('year', '')}, {top_result.get('publisher', '')}, {top_result.get('category', '')})"
            )
            print("Description:")
            print(top_result["description"])

            # Return all retrieved results so the task output can be reused if
            # this DAG is later extended.
            return {
                "query": query_str,
                "results": formatted_results,
            }

    search_vector_db_for_a_book()
    
query_data()