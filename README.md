📚 Internal Knowledge Base Bot — RAG-Powered Document Assistant
A fully local, free, privacy-first AI assistant that lets staff at any organization (law firms, construction companies, consultancies) query messy internal documents — contracts, safety regulations, project files, policies — using plain English questions, with answers grounded in the actual source documents and proper citations.
No API keys. No cloud costs. No data leaves your machine.

🎯 The Problem This Solves
Internal teams often have hundreds of PDFs, scanned contracts, and policy documents scattered across drives. Finding a specific clause, regulation, or past project detail means manually searching through dozens of files. This bot turns that pile of documents into a searchable knowledge base you can simply ask questions to.

🧠 How It Works (Architecture)
This is a Retrieval-Augmented Generation (RAG) system — instead of relying on an LLM's general knowledge, it retrieves relevant chunks from your documents first, then asks the LLM to answer based only on that retrieved context. This prevents hallucination and keeps answers grounded in real source material.

User uploads PDF/TXT
        ↓
Text extracted page-by-page (pypdf)
        ↓
Text split into overlapping chunks (800 chars, 150 overlap)
        ↓
Each chunk converted to a vector embedding (sentence-transformers)
        ↓
Embeddings stored in FAISS vector index (on disk)

─────────────────────────────────────

User asks a question
        ↓
Question converted to embedding
        ↓
FAISS finds top-N most similar chunks (semantic search)
        ↓
Retrieved chunks + question sent to local LLM (Ollama)
        ↓
LLM generates answer grounded in retrieved context
        ↓
Answer + source citations (filename, page number) returned to UI

******************************************************************************************************************************************************************************************************
gemini-rag/
├── backend/
│   ├── main.py              # FastAPI app — defines all API endpoints
│   ├── rag.py                # Core RAG logic: chunking, embedding, retrieval, generation
│   ├── ingest.py              # Document ingestion (PDF/TXT extraction)
│   ├── requirements.txt       # Python dependencies
│   ├── faiss_index/            # Persisted vector index (auto-created)
│   │   ├── index.faiss          # FAISS vector data
│   │   └── metadata.pkl          # Chunk metadata (filename, page, text)
│   └── uploads/                # Temporary storage during file upload (auto-cleaned)
├── frontend/
│   └── index.html              # Single-file chat UI (upload + query interface)
└── data/
    └── documents/                # Optional: bulk-load folder for ingestion
**********************************************************************************************************************************************************************************************************

✨ Features

Drag-and-drop document upload — PDF, TXT, or Markdown files
Automatic chunking & indexing — handles large documents by splitting into overlapping text chunks for better retrieval accuracy
Semantic search — finds relevant content even when exact keywords don't match (e.g. "termination clause" matches "ending the agreement")
Source citations — every answer shows which document (and page number, for PDFs) it came from
Knowledge base stats — see how many chunks/documents are indexed at a glance
Bulk ingestion — drop multiple files into data/documents/ and ingest them all via one API call
Persistent storage — indexed data survives backend restarts (stored in faiss_index/)
Reset endpoint — wipe the knowledge base clean when needed

**********************************************************************************************************************************************************************************************************

🚧 Known Limitations & Future Improvements

CPU inference is slow — first query can take 10–60 seconds while Ollama loads the model into memory; subsequent queries are faster
No authentication — intended for internal/trusted network use; add auth before exposing publicly
No multi-user support — single shared knowledge base, no per-user document isolation
Scanned PDFs (images) — current setup extracts text only; scanned documents need OCR (e.g. pytesseract) as a future addition
Chunk size is fixed — could be made configurable per document type for better precision

**********************************************************************************************************************************************************************************************************

📈 Possible Extensions

Swap llama3.2 for mistral or llama3.1:8b for higher-quality answers (slower)
Add OCR support for scanned contracts using pytesseract
Add conversation memory for follow-up questions
Deploy backend to a cloud VPS + frontend to Vercel/Netlify for team-wide access
Add role-based document access (e.g. HR docs vs. legal docs)
