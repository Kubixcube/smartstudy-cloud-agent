from pathlib import Path
from pypdf import PdfReader


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open or parse PDF: {pdf_path}") from exc
    pages = []

    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"Warning: failed to extract text from page {index + 1}: {exc}")
            text = ""
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