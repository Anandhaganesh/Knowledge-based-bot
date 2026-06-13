import os
import shutil
import uuid
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag import query_knowledge_base, generate_answer, get_collection_stats, reset_collection
from ingest import ingest_pdf, ingest_text_file, ingest_directory

app = FastAPI(title="Knowledge Base Bot API (Local/Free)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/")
def root():
    return {"status": "ok", "message": "Knowledge Base Bot API (Local/Free) is running"}


@app.get("/stats")
def stats():
    try:
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chunks, metadatas = query_knowledge_base(req.question, n_results=req.n_results)
        answer, sources = generate_answer(req.question, chunks, metadatas)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = (".pdf", ".txt", ".md")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.lower().endswith(".pdf"):
            chunks = ingest_pdf(file_path, file.filename)
        else:
            chunks = ingest_text_file(file_path, file.filename)

        return {
            "filename": file.filename,
            "chunks_added": chunks,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/ingest-directory")
def ingest_dir():
    docs_path = "../data/documents"
    try:
        results = ingest_directory(docs_path)
        if not results:
            return JSONResponse(
                status_code=200,
                content={"message": f"No files found in {docs_path}", "results": []}
            )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
def reset():
    try:
        reset_collection()
        return {"status": "knowledge base reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)