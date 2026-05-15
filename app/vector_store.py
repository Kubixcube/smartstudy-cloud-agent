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


def get_representative_chunks(limit: int = 12, source_file: str | None = None) -> list[dict]:
    """
    Return chunks spread across a document instead of top-k semantic matches.

    If source_file is provided, only chunks from that file are used.
    This avoids mixing multiple PDFs when generating a broad summary.
    """
    collection = get_collection()

    query = {}
    if source_file:
        query["source_file"] = source_file

    docs = list(
        collection.find(
            query,
            {
                "_id": 0,
                "text": 1,
                "source_file": 1,
                "page": 1,
                "chunk_index": 1,
                "chunk_id": 1,
            },
        ).sort([("source_file", 1), ("page", 1), ("chunk_index", 1)])
    )

    if len(docs) <= limit:
        return docs

    if limit <= 1:
        return [docs[0]]

    # Select chunks distributed from the beginning to the end of the document.
    indexes = [
        round(i * (len(docs) - 1) / (limit - 1))
        for i in range(limit)
    ]

    return [docs[i] for i in indexes]