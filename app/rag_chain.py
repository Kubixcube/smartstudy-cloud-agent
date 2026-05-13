from google import genai

from app.config import get_settings
from app.embeddings import embed_query
from app.vector_store import similarity_search
from app.tutor_prompt import build_tutor_prompt
from app.vector_store import get_representative_chunks

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
    if is_it_a_summary(question) == True : 
        print("using get_representative_chunks")
        return get_representative_chunks(limit=12) 
    query_embedding = embed_query(question)
    return similarity_search(query_embedding, k=k)

def is_it_a_summary(question: str) -> bool:
    q = question.lower()
    keywords = [
        "résumé",
        "résume",
        "resume",
        "idées principales",
        "the basics ideas",
        "overview",
        "summary",
        "main ideas",
        "main concepts",
        "vu d'ensemble",
        "concepts principaux"
    ]

    return any(keyword in q for keyword in keywords)




def answer_question(question: str, k: int = 5) -> dict:
    k = 5
    if is_it_a_summary(question) == True : k = 12
    context_chunks = retrieve_context(question, k=k)
    prompt = build_tutor_prompt(question, context_chunks)
    answer = generate_with_gemini(prompt)

    return {
        "answer": answer,
        "sources": context_chunks,
    }