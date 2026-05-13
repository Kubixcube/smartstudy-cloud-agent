import os
from pathlib import Path
from datetime import datetime, timezone

import functions_framework
from google.cloud import storage
from google import genai
from pymongo import MongoClient, UpdateOne
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "gemini-embedding-001"


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(
                {
                    "page": index + 1,
                    "text": text,
                }
            )

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")

    return pages


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


def get_genai_client():
    project = get_required_env("GOOGLE_CLOUD_PROJECT")
    location = get_required_env("GOOGLE_CLOUD_LOCATION")

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
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


def get_mongodb_collection():
    mongodb_uri = get_required_env("MONGODB_URI")
    mongodb_db = get_required_env("MONGODB_DB")
    mongodb_collection = get_required_env("MONGODB_COLLECTION")

    client = MongoClient(mongodb_uri)
    db = client[mongodb_db]
    return db[mongodb_collection]


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have the same length.")

    collection = get_mongodb_collection()
    now = datetime.now(timezone.utc)
    operations = []

    for chunk, embedding in zip(chunks, embeddings):
        document = {
            **chunk,
            "embedding": embedding,
            "updated_at": now,
        }

        operations.append(
            UpdateOne(
                {"chunk_id": chunk["chunk_id"]},
                {"$set": document},
                upsert=True,
            )
        )

    if not operations:
        return 0

    result = collection.bulk_write(operations)
    return result.upserted_count + result.modified_count


def download_blob(bucket_name: str, file_name: str, destination_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_path)


@functions_framework.cloud_event
def ingest_gcs_pdf(cloud_event):
    """
    Triggered when a PDF is uploaded to the GCS bucket.
    """
    data = cloud_event.data

    bucket_name = data.get("bucket")
    file_name = data.get("name")

    if not bucket_name or not file_name:
        print("Invalid CloudEvent payload: missing bucket or file name.")
        return

    if not file_name.lower().endswith(".pdf"):
        print(f"Skipping non-PDF file: {file_name}")
        return

    local_path = f"/tmp/{Path(file_name).name}"

    try:
        print(f"Downloading gs://{bucket_name}/{file_name} to {local_path}")
        download_blob(bucket_name, file_name, local_path)

        print("Extracting PDF text")
        pages = extract_pdf_pages(local_path)
        print(f"Extracted {len(pages)} pages")

        print("Chunking PDF")
        chunks = chunk_pdf_pages(pages, source_file=file_name)
        print(f"Created {len(chunks)} chunks")

        print("Generating embeddings")
        embeddings = embed_texts([chunk["text"] for chunk in chunks])

        print("Upserting chunks into MongoDB")
        count = upsert_chunks(chunks, embeddings)

        print(f"Successfully processed {file_name}. Upserted/modified: {count}")

    except Exception as error:
        print(f"Failed to process {file_name}: {error}")
        raise