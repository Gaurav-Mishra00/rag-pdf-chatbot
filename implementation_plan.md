# Implementation Plan - Solving RAG Chatbot Architectural Limitations & Problems

This plan details how to resolve the architectural limitations and reliability problems identified in the RAG Chatbot project:

1. **Persistent Conversational History**: Replace the in-memory dictionary history with a persistent SQLite-backed storage model.
2. **File & Metadata Tracking**: Persist uploaded PDF documents to the local filesystem and store their status/metadata in SQLite.
3. **Document Deletion Support**: Implement a `DELETE /api/v1/documents/{document_id}` endpoint that deletes the uploaded PDF, clears its metadata, and removes its indexed chunks from the FAISS vector store.
4. **Concurrency Safety**: Add thread-level write locks to FAISS index updates and persistence processes to avoid filesystem write collisions.

---

## User Review Required

> [!IMPORTANT]
> - We are introducing a lightweight SQLite database (`data/db.sqlite3`) for local storage, avoiding heavy external dependencies.
> - The FAISS index will now support deletion of documents by tracking document-to-chunk mappings in the database.
> - Uploaded PDF files will be stored in `data/uploads/` instead of being read from memory and discarded.

---

## Open Questions

- *No open questions at this stage. Standard SQLite and local directory storage meet the backend requirement without introducing third-party database engines.*

---

## Proposed Changes

### Core & Database Configuration

#### [MODIFY] [config.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/config.py)
- Add configuration settings:
  - `SQLITE_DB_PATH: str = "data/db.sqlite3"`
  - `UPLOAD_DIR: str = "data/uploads"`

#### [NEW] [database.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/database.py)
- Implement SQLite helper utility functions:
  - Context manager `get_db_connection()` to obtain connection, configure rows as dictionary-like objects, and ensure auto-commit/rollback.
  - `init_db()` to run schemas and construct table definitions (`documents`, `document_chunks`, `chat_history`).

#### [MODIFY] [main.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/main.py)
- Call `init_db()` and verify `UPLOAD_DIR` creation inside the `lifespan` startup handler.

---

### Services Layer

#### [MODIFY] [history_manager.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/services/history_manager.py)
- Refactor [HistoryManager](file:///c:/PROJECTS/rag-pdf-chatbot/app/services/history_manager.py) to run queries against the SQLite database `chat_history` table rather than using an in-memory dictionary.

---

### Vector Store Layer

#### [MODIFY] [faiss_store.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/vectorstore/faiss_store.py)
- Add thread-safe module-level lock `_faiss_write_lock = threading.Lock()` and acquire it during modification operations: `add_documents`, `delete_documents`, and `create_empty_index`.
- Modify `add_documents` to support passing custom document chunk UUIDs to map the vector index chunks to database tracking logs.
- Add `delete_documents(chunk_ids: List[str])` to delete specific vectors from FAISS using `self.vector_store.delete(ids)`.

---

### API Layer & Route Handlers

#### [MODIFY] [documents.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/api/endpoints/documents.py)
- Update `/upload` endpoint:
  1. Generate unique UUID `document_id`.
  2. Save uploaded PDF bytes to `data/uploads/{document_id}.pdf`.
  3. Pre-generate UUIDs for each split text chunk.
  4. Write metadata (`documents` and `document_chunks` mappings) to SQLite.
  5. Add chunks to FAISS using the pre-generated UUIDs.
- Add `DELETE /api/v1/documents/{document_id}` endpoint:
  1. Retrieve all chunk IDs for the document from SQLite.
  2. Remove those chunk IDs from FAISS vector store.
  3. Delete the PDF file from `data/uploads/`.
  4. Clean up SQLite metadata table rows.

---

## Verification Plan

### Automated Tests
We will update/create tests to verify:
- Persistent chat history (ensuring history doesn't reset after clearing in-memory instances).
- Document ingestion metadata persistence.
- Document deletion flow:
  1. Upload document.
  2. Perform search (verify chunk exists).
  3. Delete document.
  4. Perform search (verify chunk is removed).
- Run the full test suite:
  ```bash
  .venv\Scripts\python.exe -m pytest
  ```

### Manual Verification
- We can manually upload a document via interactive documentation `/docs`, verify status, perform a query, delete the document, and confirm search is updated.
