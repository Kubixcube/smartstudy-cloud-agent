import sys
import time
from pathlib import Path

# Make the project root importable when running:
# streamlit run ui/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from google.cloud import storage

from app.config import get_settings
from app.rag_chain import answer_question
from app.quiz import generate_quiz
from app.vector_store import get_collection
from app.memory import get_recent_history, clear_history

st.set_page_config(
    page_title="SmartStudy Cloud Agent",
    page_icon="📚",
    layout="wide",
)


def upload_pdf_to_gcs(uploaded_file, destination_name: str) -> str:
    """
    Upload a PDF file to the configured GCS bucket.
    The Cloud Function will be triggered automatically after upload.
    """
    settings = get_settings()

    if not settings.gcs_bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME is missing from the environment.")

    storage_client = storage.Client(project=settings.google_cloud_project)
    bucket = storage_client.bucket(settings.gcs_bucket_name)
    blob = bucket.blob(destination_name)

    blob.upload_from_file(
        uploaded_file,
        content_type="application/pdf",
    )

    return f"gs://{settings.gcs_bucket_name}/{destination_name}"


def count_chunks_for_file(source_file: str) -> int:
    """
    Count how many chunks are currently stored for a given source file.
    """
    collection = get_collection()
    return collection.count_documents({"source_file": source_file})


def list_indexed_files() -> list[str]:
    """
    Return the list of indexed PDF source files currently present in MongoDB.
    """
    collection = get_collection()
    files = collection.distinct("source_file")
    return sorted(files)


def display_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources were returned.")
        return

    for source in sources:
        source_file = source.get("source_file", "unknown file")
        page = source.get("page", "unknown page")
        score = source.get("score")

        if score is not None:
            st.markdown(f"- **{source_file}**, page **{page}** — score: `{score:.4f}`")
        else:
            st.markdown(f"- **{source_file}**, page **{page}**")


def wait_for_ingestion(source_file: str, max_wait_seconds: int = 90) -> int:
    """
    Poll MongoDB until chunks appear for the uploaded PDF.
    """
    progress = st.progress(0)
    status = st.empty()

    for elapsed in range(max_wait_seconds):
        count = count_chunks_for_file(source_file)

        if count > 0:
            progress.progress(100)
            status.success(f"Ingestion completed. {count} chunks found.")
            return count

        progress.progress(int((elapsed + 1) / max_wait_seconds * 100))
        status.info(
            f"Waiting for Cloud Function ingestion... "
            f"{elapsed + 1}/{max_wait_seconds}s"
        )
        time.sleep(1)

    progress.progress(100)
    status.warning(
        "No chunks were found yet. The Cloud Function may still be running. "
        "Check the Cloud Function logs if needed."
    )
    return 0


st.title("SmartStudy Cloud Agent")
st.caption("Cloud-native academic tutor for lecture PDFs")

settings = get_settings()

with st.sidebar:
    st.header("Configuration")

    st.write("**Google Cloud Project**")
    st.code(settings.google_cloud_project)

    st.write("**GCS Bucket**")
    st.code(settings.gcs_bucket_name or "Not configured")

    st.write("**MongoDB Collection**")
    st.code(f"{settings.mongodb_db}.{settings.mongodb_collection}")

    st.divider()

    st.header("Indexed files")
    try:
        indexed_files = list_indexed_files()
        if indexed_files:
            selected_file = st.selectbox(
                "Indexed PDFs",
                indexed_files,
                index=0,
            )
            st.write(f"Chunks: `{count_chunks_for_file(selected_file)}`")
        else:
            st.info("No indexed file found yet.")
    except Exception as exc:
        st.error(f"Could not read indexed files: {exc}")

tab_upload, tab_ask, tab_quiz, tab_history = st.tabs(
    ["Upload PDF", "Ask Questions", "Quiz Mode", "Conversation History"]
)


with tab_upload:
    st.subheader("Upload a lecture PDF")

    st.write(
        "Upload a PDF to Google Cloud Storage. "
        "The Cloud Function will automatically process it and store chunks in MongoDB."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    custom_name = st.text_input(
        "Destination filename in GCS",
        placeholder="example: lecture-01.pdf",
    )

    wait_after_upload = st.checkbox(
        "Wait for ingestion to complete",
        value=True,
    )

    if st.button("Upload and ingest", type="primary"):
        if uploaded_file is None:
            st.error("Please select a PDF first.")
        else:
            if custom_name.strip():
                destination_name = custom_name.strip()
                if not destination_name.lower().endswith(".pdf"):
                    destination_name += ".pdf"
            else:
                destination_name = Path(uploaded_file.name).name

            try:
                with st.spinner("Uploading PDF to Google Cloud Storage..."):
                    gcs_uri = upload_pdf_to_gcs(uploaded_file, destination_name)

                st.success("PDF uploaded successfully.")
                st.code(gcs_uri)

                st.info(
                    "The Cloud Function should now start automatically. "
                    "This may take a few seconds."
                )

                if wait_after_upload:
                    wait_for_ingestion(destination_name)

            except Exception as exc:
                st.error("Upload failed.")
                st.exception(exc)


with tab_ask:
    st.subheader("Ask questions about indexed PDFs")

    st.write(
        "Ask a question. SmartStudy retrieves the most relevant chunks from "
        "MongoDB Atlas Vector Search and answers with Gemini 2.5 Flash."
    )

    question = st.text_area(
        "Question",
        placeholder="Example: Summarize cloud-test-course.pdf",
        height=120,
    )

    k = st.slider(
        "Number of retrieved chunks",
        min_value=3,
        max_value=12,
        value=5,
    )

    if st.button("Ask SmartStudy", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving context and generating answer..."):
                    result = answer_question(question, k=k)

                st.subheader("Answer")
                st.markdown(result["answer"])

                st.subheader("Retrieved sources")
                display_sources(result["sources"])

            except Exception as exc:
                st.error("Question answering failed.")
                st.exception(exc)


with tab_quiz:
    st.subheader("Generate a quiz")

    st.write(
        "Generate a 5-question quiz from the indexed course material. "
        "You can enter a topic or a filename."
    )

    topic = st.text_input(
        "Quiz topic",
        placeholder="Example: cloud-test-course.pdf or federated learning",
    )

    quiz_k = st.slider(
        "Number of chunks used for quiz",
        min_value=3,
        max_value=12,
        value=5,
        key="quiz_k",
    )

    if st.button("Generate quiz", type="primary"):
        if not topic.strip():
            topic = "Generate a quiz from the indexed course material."

        try:
            with st.spinner("Generating quiz..."):
                result = generate_quiz(topic, k=quiz_k)

            st.subheader("Quiz")
            st.markdown(result["quiz"])

            st.subheader("Sources used")
            display_sources(result["sources"])

        except Exception as exc:
            st.error("Quiz generation failed.")
            st.exception(exc)
  
            
with tab_history:
    st.subheader("Conversation History")

    st.write(
        "This tab shows the recent conversation stored in MongoDB. "
        "It is used by SmartStudy to maintain conversational continuity."
    )

    history_limit = st.slider(
        "Number of recent messages",
        min_value=2,
        max_value=20,
        value=10,
        step=2,
    )

    if st.button("Refresh history"):
        st.rerun()

    if st.button("Clear conversation history", type="secondary"):
        deleted = clear_history()
        st.success(f"Cleared {deleted} messages from memory.")
        st.rerun()

    try:
        history = get_recent_history(limit=history_limit)

        if not history:
            st.info("No conversation history found yet.")
        else:
            for message in history:
                role = message["role"].upper()
                content = message["content"]

                if role == "USER":
                    st.markdown("### User")
                else:
                    st.markdown("### SmartStudy")

                st.markdown(content)
                st.divider()

    except Exception as exc:
        st.error("Could not load conversation history.")
        st.exception(exc)