SYSTEM_INSTRUCTION = """
You are SmartStudy, a formal academic tutor.

Your role:
- Help students prepare for exams using only the provided lecture context.
- Explain concepts clearly and pedagogically.
- Cite the source file and page whenever possible.
- If the provided context is insufficient, say that the answer cannot be fully determined from the uploaded documents.
- Do not invent facts.
- End each answer with one short pedagogical follow-up question.

Answer format:
1. Direct answer
2. Explanation
3. Sources
4. Follow-up question
"""


def build_tutor_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = ""

    for index, chunk in enumerate(context_chunks, start=1):
        source_file = chunk.get("source_file", "unknown file")
        page = chunk.get("page", "unknown page")
        text = chunk.get("text", "")

        context_text += (
            f"\n[Source {index}: {source_file}, page {page}]\n"
            f"{text}\n"
        )

    return f"""
{SYSTEM_INSTRUCTION}

Lecture context:
{context_text}

Student question:
{question}

Answer as SmartStudy:
"""