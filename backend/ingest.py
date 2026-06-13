import os
import uuid
from pypdf import PdfReader
from rag import add_document

def ingest_pdf(file_path: str, filename: str):
    """Extract text page-by-page from a PDF and ingest into vector store."""
    reader = PdfReader(file_path)
    doc_id_base = str(uuid.uuid4())[:8]
    total_chunks = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        doc_id = f"{doc_id_base}_p{page_num}"
        chunks_added = add_document(
            doc_id=doc_id,
            text=text,
            filename=filename,
            source_page=page_num
        )
        total_chunks += chunks_added

    return total_chunks


def ingest_text_file(file_path: str, filename: str):
    """Ingest a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    doc_id = str(uuid.uuid4())[:8]
    return add_document(doc_id=doc_id, text=text, filename=filename)


def ingest_directory(directory_path: str):
    """Walk a directory and ingest all supported files."""
    results = []

    if not os.path.exists(directory_path):
        return results

    for fname in os.listdir(directory_path):
        full_path = os.path.join(directory_path, fname)
        if not os.path.isfile(full_path):
            continue

        ext = fname.lower().split(".")[-1]
        try:
            if ext == "pdf":
                chunks = ingest_pdf(full_path, fname)
                results.append({"filename": fname, "chunks": chunks, "status": "success"})
            elif ext in ("txt", "md"):
                chunks = ingest_text_file(full_path, fname)
                results.append({"filename": fname, "chunks": chunks, "status": "success"})
            else:
                results.append({"filename": fname, "chunks": 0, "status": "skipped (unsupported type)"})
        except Exception as e:
            results.append({"filename": fname, "chunks": 0, "status": f"error: {str(e)}"})

    return results