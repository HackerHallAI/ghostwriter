from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance


def get_qdrant_client(
    mode: str = "local",
    host: str = "localhost",
    port: int = 6333,
    path: str = "qdrant_data",
) -> QdrantClient:
    """Initialize and return a Qdrant client with local or Docker persistence."""
    if mode == "local":
        return QdrantClient(path=path)
    elif mode == "docker":
        return QdrantClient(host=host, port=port)
    else:
        raise ValueError("Invalid mode. Choose 'local' or 'docker'.")


def ensure_collection_exists(
    client: QdrantClient, collection_name: str, vector_size: int = 0
):
    """Ensure the specified collection exists in Qdrant."""
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
