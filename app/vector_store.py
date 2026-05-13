from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne
from app.config import get_settings


def get_collection():
    settings = get_settings()
    client = MongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]
    return db[settings.mongodb_collection]


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have the same length.")

    collection = get_collection()
    operations = []
    now = datetime.now(timezone.utc)

    for chunk, embedding in zip(chunks, embeddings):
        document = {
            **chunk,
            "embedding": embedding,
            "updated_at": now,
        }

        operations.append(
            UpdateOne(
                {"chunk_id": chunk["chunk_id"]},
                {"$set": document},
                upsert=True,
            )
        )

    if not operations:
        return 0

    result = collection.bulk_write(operations)
    return result.upserted_count + result.modified_count


def similarity_search(query_embedding: list[float], k: int = 5) -> list[dict]:
    settings = get_settings()
    collection = get_collection()

    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.mongodb_vector_index,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "source_file": 1,
                "page": 1,
                "chunk_index": 1,
                "chunk_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))