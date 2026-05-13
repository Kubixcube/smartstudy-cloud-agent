import sys
from pathlib import Path

from app.pdf_loader import extract_pdf_pages
from app.chunking import chunk_pdf_pages
from app.embeddings import embed_texts
from app.vector_store import upsert_chunks


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest_local.py path/to/file.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    source_file = pdf_path.name

    print(f"[1/4] Extracting text from: {pdf_path}")
    pages = extract_pdf_pages(pdf_path)

    print(f"[2/4] Chunking {len(pages)} pages")
    chunks = chunk_pdf_pages(pages, source_file=source_file)

    print(f"[3/4] Generating embeddings for {len(chunks)} chunks")
    embeddings = embed_texts([chunk["text"] for chunk in chunks])

    print("[4/4] Upserting chunks into MongoDB Atlas")
    count = upsert_chunks(chunks, embeddings)

    print(f"Done. Upserted/modified documents: {count}")


if __name__ == "__main__":
    main()