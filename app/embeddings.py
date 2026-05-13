from google import genai
from app.config import get_settings


EMBEDDING_MODEL = "gemini-embedding-001"


def get_genai_client():
    settings = get_settings()

    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    client = get_genai_client()

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )

    return [embedding.values for embedding in response.embeddings]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]