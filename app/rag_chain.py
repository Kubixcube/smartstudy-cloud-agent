from google import genai

from app.config import get_settings
from app.embeddings import embed_query
from app.vector_store import similarity_search
from app.tutor_prompt import build_tutor_prompt


GENERATION_MODEL = "gemini-2.5-flash"


def generate_with_gemini(prompt: str) -> str:
    settings = get_settings()

    client = genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return response.text


def retrieve_context(question: str, k: int = 5) -> list[dict]:
    query_embedding = embed_query(question)
    return similarity_search(query_embedding, k=k)


def answer_question(question: str, k: int = 5) -> dict:
    context_chunks = retrieve_context(question, k=k)
    prompt = build_tutor_prompt(question, context_chunks)
    answer = generate_with_gemini(prompt)

    return {
        "answer": answer,
        "sources": context_chunks,
    }