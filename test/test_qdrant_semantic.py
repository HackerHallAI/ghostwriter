from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(url="http://localhost:6333")


hits = client.query_points(
    collection_name="web_crawled_data",
    query=encoder.encode("The Mindset Of A Digital Writer").tolist(),
    limit=3,
).points

for hit in hits:
    print(hit.payload, "score:", hit.score)
