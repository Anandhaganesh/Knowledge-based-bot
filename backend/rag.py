import os
import json
import pickle
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

FAISS_DIR = "./faiss_index"
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
META_FILE = os.path.join(FAISS_DIR, "metadata.pkl")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

os.makedirs(FAISS_DIR, exist_ok=True)

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        metadata = []
    return index, metadata


def _save_index(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
        if start >= text_len:
            break
    return chunks


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors for cosine similarity via inner product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


def add_document(doc_id: str, text: str, filename: str, source_page: int = None):
    """Chunk a document, embed it, and add to the FAISS index."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embedder = get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = _normalize(embeddings.astype("float32"))

    index, metadata = _load_index()
    index.add(embeddings)

    for i, chunk in enumerate(chunks):
        metadata.append({
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk,
            "filename": filename,
            "chunk_index": i,
            "source_page": source_page if source_page is not None else -1
        })

    _save_index(index, metadata)
    return len(chunks)


def query_knowledge_base(query: str, n_results: int = 5):
    """Retrieve relevant chunks for a query."""
    index, metadata = _load_index()

    if index.ntotal == 0:
        return [], []

    n_results = min(n_results, index.ntotal)

    embedder = get_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_vec = _normalize(query_vec.astype("float32"))

    scores, indices = index.search(query_vec, n_results)

    documents = []
    metadatas = []
    for idx in indices[0]:
        if idx == -1 or idx >= len(metadata):
            continue
        item = metadata[idx]
        documents.append(item["text"])
        metadatas.append({
            "filename": item["filename"],
            "chunk_index": item["chunk_index"],
            "source_page": item["source_page"]
        })

    return documents, metadatas


def generate_answer(query: str, context_chunks: list, metadatas: list):
    """Generate an answer using retrieved context via local Ollama model."""
    if not context_chunks:
        return (
            "I don't have any documents in the knowledge base yet, "
            "or I couldn't find relevant information to answer this question.",
            []
        )

    context_blocks = []
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):
        source = meta.get("filename", "unknown")
        page = meta.get("source_page", -1)
        page_info = f", page {page}" if page and page != -1 else ""
        context_blocks.append(f"[Source {i+1}: {source}{page_info}]\n{chunk}")

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "You are an internal knowledge base assistant for a business. "
        "Answer the user's question using ONLY the provided context below. "
        "If the context does not contain enough information to answer, "
        "say so clearly instead of guessing. "
        "Always cite which source you used, e.g. [Source 1].\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        if not answer:
            answer = "The local model returned an empty response. Try rephrasing your question."
    except requests.exceptions.ConnectionError:
        answer = (
            "⚠️ Could not connect to Ollama. Make sure Ollama is running "
            "(open a terminal and run: ollama serve) and that you've pulled "
            f"the model with: ollama pull {OLLAMA_MODEL}"
        )
    except Exception as e:
        answer = f"⚠️ Error generating answer: {str(e)}"

    sources = []
    seen = set()
    for meta in metadatas:
        fname = meta.get("filename", "unknown")
        page = meta.get("source_page", -1)
        key = (fname, page)
        if key not in seen:
            seen.add(key)
            sources.append({"filename": fname, "page": page})

    return answer, sources


def get_collection_stats():
    index, metadata = _load_index()

    if index.ntotal == 0:
        return {"total_chunks": 0, "documents": []}

    filenames = set()
    for item in metadata:
        filenames.add(item.get("filename", "unknown"))

    return {
        "total_chunks": index.ntotal,
        "documents": sorted(list(filenames))
    }


def reset_collection():
    """Delete the FAISS index and metadata files."""
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(META_FILE):
        os.remove(META_FILE)
    return True