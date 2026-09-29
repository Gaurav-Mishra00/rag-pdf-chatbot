# Step-by-Step Technical Build Guide: RAG PDF Chatbot

This document provides a comprehensive, step-by-step technical guide for building the **RAG PDF Chatbot**. It explains the architecture, frameworks, database design, vector database integration, embedding models, LLM APIs, and security measures used throughout the application.

---

## 1. System Overview & Architecture

The application is an enterprise-grade Retrieval-Augmented Generation (RAG) system designed to parse uploaded PDF documents, extract text and page metadata, split text into semantic chunks, generate dense vector embeddings, and answer conversational user questions with strict grounding and verifiable citations.

### Core Architecture Flow

```
[ User PDF Upload ]
       │
       ▼
[ FastAPI /api/v1/documents/upload ] ──> Validate Magic Bytes (%PDF-) & File Size (<=10MB)
       │
       ├─> Store PDF in data/uploads/{uuid}.pdf
       ├─> Insert metadata into SQLite documents table
       │
       ▼
[ PDFProcessorService (PyMuPDF) ] ──> Extract text per page with page & source metadata
       │
       ▼
[ RecursiveCharacterTextSplitter ] ──> 1000-char chunks, 200 overlap
       │
       ▼
[ Embedding Model (BAAI/bge-m3) ] ──> Dense 1024-dim unit-normalized vectors
       │
       ▼
[ FAISS Vector Store ] ──> Save IndexFlatIP vectors + JSON manifest under process RLock
```

```
[ User Chat Question ]
       │
       ▼
[ FastAPI /api/v1/chat/query ] ──> Verify API Key (SHA-256 Tenant Hash) & Rate Limiting
       │
       ▼
[ HistoryManager ] ──> Fetch recent session messages from SQLite chat_history table
       │
       ▼
[ Query Rewriter ] ──> Condense follow-up questions into standalone search query
       │
       ▼
[ FAISS Cosine Retrieval ] ──> Retrieve Top-K candidate chunks filtered by user_id
       │
       ▼
[ CrossEncoder Reranker ] ──> Score query-chunk pairs (ms-marco-MiniLM-L-6-v2)
       │
       ▼
[ Relevance Gating ] ──> If max_score < RELEVANCE_THRESHOLD:
       │                   └─> Return "I don't know" immediately (Skip LLM call)
       │
       ▼
[ Grounded LLM Generation ] ──> Invoke Gemini / OpenAI / Anthropic with [Source X] citations
       │
       ▼
[ Response & Persistence ] ──> Save turn to SQLite, return answer, citations, and timings
```

---

## 2. Technology Stack & Component Justification

| Layer | Technology | Version | Purpose & Technical Justification |
| :--- | :--- | :--- | :--- |
| **Web Framework** | **FastAPI** | `>=0.115` | Modern asynchronous ASGI framework; provides native Pydantic validation, dependency injection, OpenAPI documentation, and high concurrency. |
| **ASGI Server** | **Uvicorn** | `>=0.30` | Production ASGI server implementation based on `uvloop` and `httptools`. |
| **Relational Database** | **SQLite** (`sqlite3`) | Built-in | Zero-dependency, embedded ACID database. Configured with Write-Ahead Logging (`WAL`) and busy timeouts for high concurrent throughput without external servers. |
| **Vector Store** | **Native FAISS** (`faiss-cpu`) | `>=1.8` | High-performance dense vector index (`IndexFlatIP`). Wrapped with unit normalization for cosine similarity and JSON manifests to avoid Python pickle vulnerabilities. |
| **Embedding Model** | **BAAI/bge-m3** | via `sentence-transformers` | Leading open-weight multi-lingual embedding model (1024 dimensions, 8192 token context window) providing dense representation. |
| **Reranker** | **cross-encoder/ms-marco-MiniLM-L-6-v2** | via `sentence-transformers` | Cross-encoder architecture that jointly processes query and passage, outputting calibrated relevance scores via a sigmoid transfer function. |
| **LLM Provider APIs** | **Google Gemini** / **OpenAI** / **Anthropic** | via `langchain-*` | Primary: `gemma-4-26b-a4b-it` / `gemini-1.5-flash` via `ChatGoogleGenerativeAI`. Configurable to OpenAI (`gpt-4o-mini`) or Anthropic (`claude-3-5-haiku`). |
| **PDF Processing** | **PyMuPDF (`fitz`)** | `>=1.24` | Significantly faster than `pypdf` or `pdfminer`; extracts clean text per page while preserving document metadata. |
| **Configuration** | **Pydantic Settings** | `>=2.4` | Enforces type validation on `.env` variables, supports file-based Docker/K8s secrets. |
| **Frontend UI** | **Vanilla HTML5 / CSS / ES6+ JS** | Native | Single-page application served directly from FastAPI static files; responsive dark-mode UI with no build tools needed. |
| **Testing** | **pytest**, `pytest-asyncio`, `httpx` | Latest | 95 automated unit, architectural, failure handling, and multi-tenant test cases. |

---

## 3. Step-by-Step Implementation Guide

### Step 1: Project Structure Setup

Create a modular directory structure that separates API routing, business services, core configuration, and vector storage:

```bash
mkdir -p app/api/endpoints app/core app/prompts app/schemas app/services app/static app/vectorstore data/backups data/faiss_index data/uploads docs scripts tests
```

---

### Step 2: Configuration & Environment Setup (`app/core/config.py`)

Implement configuration management using Pydantic's `BaseSettings`:
- Define strongly-typed settings for LLM providers, embedding models, timeouts, file limits, and directories.
- Add automatic file-based secret resolution (`resolve_secret_file`) to support Docker and Kubernetes secret volume mounts.

Key Configuration Properties:
```python
class Settings(BaseSettings):
    APP_NAME: str = "FastAPI RAG Chatbot"
    APP_ENV: str = "development"
    API_KEY: str = "change_me_in_production"
    
    # LLM Provider Configuration
    LLM_PROVIDER: str = "google"  # google | openai | anthropic
    LLM_MODEL_NAME: str = "gemma-4-26b-a4b-it"
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SECONDS: float = 30.0
    
    # Embedding Configuration
    EMBEDDINGS_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    
    # Reranking & Relevance Configuration
    RERANK_ENABLED: bool = True
    RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RELEVANCE_THRESHOLD: float = 0.35
    
    # Storage Paths
    SQLITE_DB_PATH: str = "data/db.sqlite3"
    UPLOAD_DIR: str = "data/uploads"
    FAISS_INDEX_PATH: str = "data/faiss_index"
```

---

### Step 3: SQLite Database & WAL Concurrency (`app/core/database.py`)

To eliminate the risk of data loss from in-memory history without introducing heavy external database engines, use SQLite with Write-Ahead Logging:

1. **Connection Manager**:
   ```python
   @contextmanager
   def get_db_connection():
       conn = sqlite3.connect(settings.SQLITE_DB_PATH, timeout=10.0)
       conn.row_factory = sqlite3.Row
       conn.execute("PRAGMA journal_mode=WAL")
       conn.execute("PRAGMA busy_timeout=5000")
       conn.execute("PRAGMA synchronous=NORMAL")
       try:
           yield conn
           conn.commit()
       except Exception:
           conn.rollback()
           raise
       finally:
           conn.close()
   ```

2. **Database Schemas**:
   - `documents`: Stores `document_id` (UUID), `filename`, `file_path`, `file_size_bytes`, `chunk_count`, `status`, and `user_id`.
   - `document_chunks`: Maps `chunk_id` to `document_id` for deletion cascades.
   - `chat_history`: Stores `id`, `session_id`, `role` (`user`/`assistant`), `content`, `user_id`, and `created_at`.
   - Create indices on `(user_id, created_at)` and `(session_id, user_id)` for sub-millisecond retrieval.

3. **Storage Reconciliation**:
   Implement `reconcile_storage_layers()` to identify and automatically prune orphaned files, dangling DB records, and orphaned vector chunks on application startup.

---

### Step 4: PDF Extraction & Chunking (`app/services/pdf_processor.py`)

1. **Extraction**: Use PyMuPDF (`fitz`) to iterate through pages, extracting text and attaching metadata (`page_number`, `total_pages`, `filename`).
2. **Chunking**: Use `RecursiveCharacterTextSplitter` configured with:
   - `chunk_size = 1000`
   - `chunk_overlap = 200`
   - Separators: `["\n\n", "\n", " ", ""]`
3. Attach unique UUIDs to every chunk and record parent document associations.

---

### Step 5: Native FAISS Vector Store Adapter (`app/vectorstore/native_faiss.py` & `faiss_store.py`)

Rather than relying on deprecated and vulnerable pickle-based LangChain FAISS wrappers, build an application-owned FAISS adapter:

1. **Index Initialization**:
   Use `faiss.IndexFlatIP(dimension)`.
2. **Cosine Similarity**:
   Unit-normalize all vectors using `faiss.normalize_L2(matrix)` prior to adding or searching. In an inner-product index, normalized vectors produce mathematically exact cosine similarity:
   $$\text{cosine\_similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \hat{u} \cdot \hat{v} = \text{IP}(\hat{u}, \hat{v})$$
3. **Safe Storage**:
   Persist vectors as binary FAISS format and document metadata as JSON (`manifest.json`), eliminating Python pickle deserialization risks.
4. **Thread Safety**:
   Wrap index writes (`add_documents`, `delete_documents`, `save_index`) with a process-level `threading.RLock()`.
5. **Multi-Tenant Scoping**:
   Filter retrieved chunks during search to ensure `chunk.metadata["user_id"] == current_user_id`.

---

### Step 6: Multi-Stage RAG Pipeline (`app/services/rag_service.py`)

Implement the complete conversational RAG execution flow:

1. **Conversational Question Condensation**:
   If history exists, format past turns and user prompt into `CONDENSE_QUESTION_PROMPT`. Have the LLM return a standalone search query.
2. **Dense Retrieval**:
   Retrieve top $K$ candidate chunks (e.g. $K=8$) from FAISS.
3. **Cross-Encoder Reranking (`app/services/reranker.py`)**:
   Pass $(q, \text{chunk})$ pairs to `ms-marco-MiniLM-L-6-v2`. Transform logits via sigmoid function:
   $$\text{score} = \frac{1}{1 + e^{-\text{logit}}}$$
   Sort documents by rerank score descending.
4. **Relevance Gating ("I don't know" circuit breaker)**:
   If top score $< \text{RELEVANCE\_THRESHOLD}$ (0.35), bypass LLM generation completely and return:
   > *"I do not have sufficient information in the uploaded documents to answer this question."*
5. **Grounded Generation**:
   Construct system prompt:
   - Provide structured contexts labeled `[Source 1]`, `[Source 2]`, etc.
   - Instruct model to only answer using provided context and cite inline.
6. **Citation Resolution**:
   Parse citations in model response text, map to source document name, page, and chunk snippet, and format response metadata.

---

### Step 7: Persistent Conversation History (`app/services/history_manager.py`)

Implement the asynchronous history manager:
- Offload SQLite queries to thread pools via `anyio.to_thread.run_sync`.
- Provide `get_history(session_id, user_id)`, `add_message(session_id, role, content, user_id)`, `list_sessions(user_id)`, and `clear_history(session_id, user_id)`.

---

### Step 8: Multi-Tenancy, Security & Rate Limiting (`app/core/security.py`, `rate_limiter.py`)

1. **Authentication**:
   Inspect incoming `X-API-Key` headers. Hash the key with SHA-256 to create a deterministic `user_id`.
2. **Rate Limiting**:
   Implement in-memory sliding window token bucket limiters for chat (`check_chat_rate_limit`) and upload (`check_upload_rate_limit`).
3. **Content Security Policy (CSP)**:
   Implement FastAPI middleware injecting strict headers (`default-src 'self'`). For `/docs` and `/redoc`, generate dynamic per-request cryptographic nonces allowing Swagger CDN scripts without weakening application-wide CSP.

---

### Step 9: Diagnostics, Operations & Backup (`app/core/backup.py`, `app/main.py`)

1. **Readiness Probe (`/health/ready`)**:
   Tests SQLite database connectivity and FAISS index readiness. Returns HTTP 200 when ready, HTTP 503 if any component is degraded.
2. **Liveness Probe (`/health`)**:
   Returns HTTP 200 to confirm ASGI process liveness.
3. **Automated Snapshots**:
   On startup, copy the SQLite database and FAISS directory into a timestamped ZIP archive in `data/backups/`, rotating older backups beyond the retention limit.

---

### Step 10: Frontend Web Interface (`app/static/`)

Build a clean, responsive single-page web UI:
- **`index.html`**: Clean semantic HTML structure with navigation sidebar, central chat workspace, and source library panel.
- **`styles.css`**: CSS variables for dark/light themes, ambient lighting, responsive flexbox/grid layout, and subtle micro-animations.
- **`app.js`**: Vanilla ES6 JavaScript handling asynchronous API requests, document drag-and-drop upload, message rendering, citation badges, and session state persistence.

---

### Step 11: Production Containerization (`Dockerfile`)

Create a two-stage Dockerfile to minimize image size and exclude build tools from the final production container:

```dockerfile
# Stage 1: Builder
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.lock .
RUN pip install --no-cache-dir --user -r requirements.lock

# Stage 2: Runner
FROM python:3.12-slim AS runner
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /root/.local
COPY . /app
RUN mkdir -p /app/data/backups /app/data/uploads /app/data/faiss_index
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 4. Verification and Testing

Run the automated test suite to verify all architectural requirements:

```bash
pytest
```

All **95 tests** validate:
- Persistent chat history across server reloads.
- Concurrency write locking in FAISS.
- PDF upload validation (magic bytes, size limits).
- Document deletion and vector chunk removal.
- Provider error mapping (502, 503, 504) and non-persistence of failed turns.
- Multi-tenant privacy boundaries and rate limiting.
- Cross-encoder reranking and relevance thresholding.
- Self-healing storage reconciliation.
- Swagger and ReDoc documentation CSP nonces.
