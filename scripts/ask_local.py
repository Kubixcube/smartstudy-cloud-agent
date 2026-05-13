import sys

from app.rag_chain import answer_question


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.ask_local "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    result = answer_question(question)

    print("\n=== SmartStudy Answer ===\n")
    print(result["answer"])

    print("\n=== Retrieved Sources ===\n")
    for source in result["sources"]:
        print(
            f"- {source['source_file']}, page {source['page']}, "
            f"score={source.get('score', 0):.4f}"
        )


if __name__ == "__main__":
    main()