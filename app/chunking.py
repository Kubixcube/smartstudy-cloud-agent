from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pdf_pages(
    pages: list[dict],
    source_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "text": chunk_text,
                    "source_file": source_file,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "chunk_id": f"{source_file}_p{page_number}_c{chunk_index}",
                }
            )

    return chunks