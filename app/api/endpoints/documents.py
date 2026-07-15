import logging
import os
import uuid
from typing import List, Optional, Tuple

import anyio
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status

from app.api.deps import get_pdf_processor, get_vector_store
from app.core.config import settings
from app.core.database import get_db_connection
from app.core.security import verify_api_key
from app.schemas.document import DocumentUploadResponse, DocumentStatusResponse, IngestionStatus
from app.services.pdf_processor import PDFProcessorService
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------
# Database Query Helpers (Sync functions for run_sync)
# ------------------------------------------------------------------

def _db_check_document_exists(filename: str) -> bool:
    with get_db_connection() as conn:
        row = conn.execute("SELECT 1 FROM documents WHERE filename = ?", (filename,)).fetchone()
        return row is not None


def _db_insert_document(
    doc_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    chunk_count: int,
    status: str,
) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (document_id, filename, file_path, file_size_bytes, chunk_count, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_id, filename, file_path, file_size, chunk_count, status),
        )


def _db_insert_chunks(doc_id: str, chunk_ids: List[str]) -> None:
    with get_db_connection() as conn:
        for chunk_id in chunk_ids:
            conn.execute(
                "INSERT INTO document_chunks (chunk_id, document_id) VALUES (?, ?)",
                (chunk_id, doc_id),
            )


def _db_delete_document_records(document_id: str) -> None:
    with get_db_connection() as conn:
        # Cascade delete is enabled, deleting from documents automatically clears document_chunks
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def _db_get_document_details(document_id: str) -> Tuple[str, List[str]] | None:
    with get_db_connection() as conn:
        doc_row = conn.execute(
            "SELECT file_path FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not doc_row:
            return None
        file_path = doc_row["file_path"]

        chunk_rows = conn.execute(
            "SELECT chunk_id FROM document_chunks WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        chunk_ids = [row["chunk_id"] for row in chunk_rows]
        return file_path, chunk_ids


def _db_list_documents() -> List[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT document_id, filename, file_size_bytes, chunk_count, status, error_message, created_at FROM documents"
        ).fetchall()
        return [dict(row) for row in rows]


def _db_get_document_by_id(document_id: str) -> Optional[dict]:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT document_id, filename, file_size_bytes, chunk_count, status, error_message, created_at "
            "FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return dict(row) if row else None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get(
    "",
    response_model=List[DocumentStatusResponse],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def list_documents() -> List[DocumentStatusResponse]:
    """
    Lists all documents stored in the database metadata table.
    Returns a typed list of DocumentStatusResponse objects.
    """
    rows = await anyio.to_thread.run_sync(_db_list_documents)
    return [DocumentStatusResponse(**row) for row in rows]


@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def get_document(
    document_id: str,
) -> DocumentStatusResponse:
    """
    Returns full metadata for a single document by its document_id.
    Raises 404 if the document does not exist.
    """
    row = await anyio.to_thread.run_sync(_db_get_document_by_id, document_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return DocumentStatusResponse(**row)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
)
async def upload_document(
    file: UploadFile = File(...),
    pdf_processor: PDFProcessorService = Depends(get_pdf_processor),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """
    Uploads a PDF file, saves it locally, parses/chunks its text content,
    generates embeddings with FAISS, and stores mapping metadata in SQLite.
    Supports transactional rollback/cleanup on failure.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Prevent duplicates by filename
    exists = await anyio.to_thread.run_sync(_db_check_document_exists, file.filename)
    if exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document with filename '{file.filename}' already exists.",
        )

    # 1. Read uploaded file bytes
    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard limit
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read file: {str(e)}",
        )

    # 1a. Enforce file size limit
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of 50 MB.",
        )

    # 1b. Magic-byte validation: all valid PDFs start with the 4-byte signature %PDF
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid PDF (missing %PDF header).",
        )

    # 2. Setup document IDs and paths
    document_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}.pdf")

    # 3. Save uploaded PDF physically to the file system
    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save PDF to disk: {str(e)}",
        )

    # 4. Extract and chunk text from the PDF
    documents = pdf_processor.process_pdf(content, filename=file.filename)
    if not documents:
        # Cleanup file if parsing failed or PDF is empty
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse text from the PDF file or document was empty.",
        )

    # 5. Pre-generate chunk UUIDs for vector mapping
    chunk_ids = [str(uuid.uuid4()) for _ in documents]

    # 6. Insert metadata records in SQLite DB
    try:
        await anyio.to_thread.run_sync(
            _db_insert_document,
            document_id,
            file.filename,
            file_path,
            len(content),
            len(documents),
            IngestionStatus.COMPLETED.value,
        )
        await anyio.to_thread.run_sync(_db_insert_chunks, document_id, chunk_ids)
    except Exception as e:
        # Rollback file write
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store metadata in database: {str(e)}",
        )

    # 7. Add chunks to the FAISS index
    try:
        vector_store.add_documents(documents, ids=chunk_ids)
    except Exception as e:
        # Rollback: Clean database records & delete physical file
        await anyio.to_thread.run_sync(_db_delete_document_records, document_id)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add documents to FAISS index: {str(e)}",
        )

    return DocumentUploadResponse(
        filename=file.filename,
        status=IngestionStatus.COMPLETED,
        message=f"Successfully parsed, stored, and indexed {len(documents)} chunks.",
        document_id=document_id,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def delete_document(
    document_id: str,
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> dict:
    """
    Deletes an uploaded PDF document: removes its chunks from FAISS,
    removes the local file from disk, and purges its metadata from SQLite database.
    """
    details = await anyio.to_thread.run_sync(_db_get_document_details, document_id)
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    file_path, chunk_ids = details

    # Deletion order and partial-failure story:
    # Step 1 — FAISS deletion: if this fails, abort; DB and disk are untouched (safe to retry).
    # Step 2 — Disk deletion: best-effort; log and continue even on error.
    # Step 3 — DB deletion: if this fails AFTER FAISS has already been mutated, the FAISS
    #           deletion cannot be undone.  We log the orphaned chunk IDs prominently so an
    #           operator can manually clean the DB, and return a 500 with full detail.

    # 1. Delete chunks from FAISS index (abort whole operation on failure)
    try:
        if chunk_ids:
            vector_store.delete_documents(chunk_ids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove chunks from vector store index: {str(e)}",
        )

    # 2. Delete the physical saved PDF file (best-effort; log on error, do not abort)
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(
            "Could not delete physical file '%s' for document '%s': %s — manual cleanup required.",
            file_path, document_id, e,
        )

    # 3. Clean database metadata entries (cascade deletes document_chunks rows too)
    try:
        await anyio.to_thread.run_sync(_db_delete_document_records, document_id)
    except Exception as e:
        # FAISS deletion has already committed and cannot be rolled back.
        # Log orphaned IDs so an operator can manually DELETE FROM document_chunks.
        logger.critical(
            "DB cleanup failed for document '%s' after FAISS deletion succeeded. "
            "Orphaned chunk IDs: %s — manual DB cleanup required: "
            "DELETE FROM document_chunks WHERE document_id='%s'; "
            "DELETE FROM documents WHERE document_id='%s'.",
            document_id, chunk_ids, document_id, document_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Vector store was updated but database cleanup failed: {str(e)}. "
                f"Document ID '{document_id}' may need manual DB cleanup."
            ),
        )

    return {
        "message": "Document successfully deleted.",
        "document_id": document_id,
    }
