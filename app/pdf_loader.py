from pathlib import Path
from pypdf import PdfReader


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

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