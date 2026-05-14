from google import genai

from app.config import get_settings
from app.embeddings import embed_query
from app.vector_store import similarity_search


GENERATION_MODEL = "gemini-2.5-flash"


def generate_quiz(topic: str, k: int = 5) -> dict:
    settings = get_settings()

    query_embedding = embed_query(topic)
    context_chunks = similarity_search(query_embedding, k=k)

    context_text = "\n\n".join(
        [
            f"[Source: {chunk['source_file']} - page {chunk['page']}]\n"
            f"{chunk['text']}"
            for chunk in context_chunks
        ]
    )

    prompt = f"""
        You are SmartStudy, an educational AI tutor.

        Using ONLY the provided lesson context, generate a 5-question quiz.

        Requirements:
        - Questions must test understanding of the material.
        - Include a mix of easy and medium questions.
        - Include the correct answer after each question.
        - Format clearly in Markdown.
        - Do NOT invent information outside the context.

        Lesson Context:
        {context_text}
        """

    client = genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return {
        "quiz": response.text,
        "sources": context_chunks,
    }