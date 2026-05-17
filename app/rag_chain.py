from google import genai

from app.config import get_settings
from app.embeddings import embed_query
from app.vector_store import similarity_search, get_representative_chunks
from app.tutor_prompt import build_tutor_prompt

from app.memory import (
    save_message,
    get_recent_history,
    format_history_for_prompt,
)


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


def is_summary_request(question: str) -> bool:
    q = question.lower()

    keywords = [
        "résumé",
        "résume",
        "resume",
        "summarize",
        "summary",
        "overview",
        "main idea",
        "main ideas",
        "main concept",
        "main concepts",
        "idées principales",
        "concepts principaux",
        "vue d'ensemble",
        "vu d'ensemble",
    ]

    return any(keyword in q for keyword in keywords)


def extract_source_file_from_question(question: str) -> str | None:
    """
    Very simple detection of a PDF filename mentioned in the question.

    Example:
    'Summarize cloud-test-course.pdf' -> 'cloud-test-course.pdf'
    """
    words = question.replace('"', " ").replace("'", " ").split()

    for word in words:
        cleaned = word.strip(".,;:()[]{}")
        if cleaned.lower().endswith(".pdf"):
            return cleaned

    return None


def retrieve_context(question: str, k: int = 5) -> list[dict]:
    if is_summary_request(question):
        source_file = extract_source_file_from_question(question)
        return get_representative_chunks(limit=12, source_file=source_file)

    query_embedding = embed_query(question)
    return similarity_search(query_embedding, k=k)


def build_memory_prompt(history_text: str) -> str:
    return f"""
            Previous conversation history:
            {history_text}

            For factual questions about lessons or documents, use the retrieved lesson context
            as the primary source of truth.

            If the answer exists in the conversation history but not in the lesson documents,
            you are allowed to answer using the conversation history but you must clearly indicate
            that the information is from the conversation history and not the lesson context.
            """


def answer_question(question: str, k: int = 5, session_id: str = "default") -> dict:
    history = get_recent_history(session_id)
    history_text = format_history_for_prompt(history)

    context_chunks = retrieve_context(question, k=k)

    base_prompt = build_tutor_prompt(question, context_chunks)

    memory_prompt = build_memory_prompt(history_text)

    final_prompt = f"""
                    {memory_prompt}

                    {base_prompt}
                    """

    answer = generate_with_gemini(final_prompt)

    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return {
        "answer": answer,
        "sources": context_chunks,
        "history_used": history,
    }