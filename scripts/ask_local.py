import sys

from app.rag_chain import answer_question
from app.quiz import generate_quiz


def print_sources(sources: list[dict]) -> None:
    print("\n=== Retrieved Sources ===\n")
    for source in sources:
        print(
            f"- {source['source_file']}, page {source['page']}, "
            f"score={source.get('score', 0):.4f}"
        )


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.ask_local "your question"')
        print('Quiz:  python -m scripts.ask_local "/quiz your topic"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    if question.strip().lower().startswith("/quiz"):
        topic = question.strip()[len("/quiz"):].strip()
        if not topic:
            topic = "Generate a quiz from the indexed course material."

        result = generate_quiz(topic)

        print("\n=== SmartStudy Quiz ===\n")
        print(result["quiz"])
        print_sources(result["sources"])
        return

    result = answer_question(question)

    print("\n=== SmartStudy Answer ===\n")
    print(result["answer"])
    print_sources(result["sources"])


if __name__ == "__main__":
    main()