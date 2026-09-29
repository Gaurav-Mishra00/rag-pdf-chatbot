# Codebase Structure and File Catalog

This document provides a complete, file-by-file breakdown of the **RAG PDF Chatbot** codebase after cleanup. It details the purpose, internal functions, and interactions of every module in the application.

---

## 1. Directory Overview

```text
rag-pdf-chatbot/
├── app/
│   ├── api/                     # Web layer: routes, dependency injection, and endpoints
│   │   ├── endpoints/           # Individual feature route handlers
│   │   │   ├── chat.py          # Conversational query and evaluation endpoints
│   │   │   ├── documents.py     # PDF upload, listing, status, and deletion endpoints
│   │   │   ├── sessions.py      # Session management and message history retrieval
│   │   │   └── vectorstore.py   # Direct vector store search, status, and tally endpoints
│   │   ├── deps.py              # Centralized dependency injection factories & singletons
│   │   └── router.py            # Master APIRouter aggregating all sub-routers under /api/v1
│   ├── core/                    # Infrastructure and system-level cross-cutting concerns
│   │   ├── backup.py            # Automated snapshot backups of SQLite DB & FAISS indices
│   │   ├── cache.py             # Thread-safe in-memory TTL query cache with LRU eviction
│   │   ├── config.py            # Pydantic Settings configuration loader (.env)
│   │   ├── database.py          # SQLite connection manager, schema DDL, & self-healing reconciliation
│   │   ├── rate_limiter.py      # In-memory sliding-window rate limiters for chat & upload
│   │   └── security.py          # API key verification & SHA-256 tenant hash generator
│   ├── prompts/                 # Prompt engineering and template strings
│   │   └── templates.py         # Question rewriter and document-grounded QA prompt templates
│   ├── schemas/                 # Pydantic data transfer objects (DTOs) and API contracts
│   │   ├── chat.py              # Request/response models for chat, citations, timings, & metrics
│   │   └── document.py          # Request/response models for PDF uploads, status, & search
│   ├── services/                # Business logic implementation
│   │   ├── errors.py            # Categorized error types for safe provider exception handling
│   │   ├── history_manager.py   # SQLite-backed persistent conversation history service
│   │   ├── pdf_processor.py     # Text extraction (PyMuPDF) and chunking service
│   │   ├── rag_service.py       # Full RAG pipeline orchestrator (rewrite -> retrieve -> rerank -> gate -> generate)
│   │   └── reranker.py          # Lazy process-wide CrossEncoder reranker with sigmoid confidence
│   ├── static/                  # Frontend single-page application assets
│   │   ├── index.html           # Modern responsive HTML5 UI (Atlas)
│   │   ├── styles.css           # Custom CSS styling (dark/light themes, ambient lighting, responsive cards)
│   │   └── app.js               # Vanilla JS client logic for chatting, uploading, and citations
│   ├── vectorstore/             # Vector database integration
│   │   ├── faiss_store.py       # High-level FAISSVectorStore with thread locks & tenant filtering
│   │   └── native_faiss.py      # Native FAISS adapter using JSON manifests (no Python pickle)
│   └── main.py                  # FastAPI application entrypoint, middleware, docs CSP, & lifecycles
├── data/                        # Persistent application storage
│   ├── backups/                 # Timestamped zip archives of SQLite and FAISS data
│   ├── faiss_index/             # Native FAISS index files (manifest.json, vectors.faiss)
│   ├── uploads/                 # Uploaded PDF files stored on disk named by UUID
│   └── db.sqlite3               # SQLite database file
├── docs/                        # Project technical documentation and guides
│   ├── architecture.md          # High-level flow diagrams and architecture documentation
│   ├── codebase-structure-and-guide.md # This complete file-by-file catalog
│   ├── problems-and-solutions.md# Comprehensive problems, root causes, solutions, & trade-offs
│   └── technical-build-guide.md # Step-by-step technical implementation guide
├── scripts/                     # Standalone operational tools
│   └── reindex.py               # CLI tool to rebuild FAISS index from disk PDFs & SQLite metadata
├── tests/                       # Automated test suite (110 pytest tests)
│   ├── conftest.py              # Global fixtures, mock providers, and isolated test databases
│   ├── test_api.py              # Basic endpoint smoke and routing tests
│   ├── test_arch_problems.py    # Architectural regression tests (WAL mode, locks, deletion)
│   ├── test_chat_failures.py    # Error handling, provider timeouts, and generation failures
│   ├── test_dependency_cache_and_docs.py # Concurrency caching and documentation CSP nonces
│   ├── test_e2e.py              # Full end-to-end upload and chat verification
│   ├── test_efficiency.py       # QueryCache, chunk deduplication, HNSW index, and SSE streaming
│   ├── test_missing_apis.py     # Session listing, pagination, and readiness probe tests
│   ├── test_multi_tenant.py     # Per-user data isolation and tenant boundary tests
│   ├── test_operational.py      # Rate limiting, payload limits, backups, and secret files
│   ├── test_p2_polish.py        # Reranker, relevance gating, and path masking tests
│   ├── test_reconciliation.py   # Self-healing storage reconciliation tests
│   ├── test_security_hardening.py # CORS, HSTS, timing safety, and upload traversal tests
│   └── test_services.py         # Unit tests for PDF processor, FAISS store, & RAG service
├── .env.example                 # Reference environment variables
├── Dockerfile                   # Two-stage production container build
├── requirements.txt             # Primary Python dependency specifications
└── requirements.lock            # Fully pinned Python dependencies
```

---

## 2. File-by-File Breakdown

### App Core (`app/core/`)

#### [config.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/config.py)
- **Role**: Centralized configuration management using Pydantic's `BaseSettings`.
- **What it does**:
  - Reads environment variables from `.env` or system environment.
  - Declares strongly-typed configuration parameters: `APP_NAME`, `API_KEY`, `LLM_PROVIDER` (Google, OpenAI, Anthropic), `LLM_MODEL_NAME`, `EMBEDDINGS_PROVIDER`, `EMBEDDING_MODEL_NAME`, `SQLITE_DB_PATH`, `UPLOAD_DIR`, `FAISS_INDEX_PATH`, chunk sizes, timeouts, and rate limits.
  - Automatically resolves file-based secrets (`*_FILE` patterns) commonly used in Docker and Kubernetes secrets.

#### [database.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/database.py)
- **Role**: Database connection manager and self-healing storage reconciliation.
- **What it does**:
  - `get_db_connection()`: Context manager that yields an active SQLite connection configured with `PRAGMA journal_mode=WAL` (Write-Ahead Logging) and `PRAGMA busy_timeout=5000` to prevent database locking during concurrent writes.
  - `init_db()`: Executes DDL schemas to initialize three core tables:
    1. `documents`: Tracks document ID, filename, disk file path, size, chunk count, upload status, and owner `user_id`.
    2. `document_chunks`: Maps chunk IDs to parent `document_id`.
    3. `chat_history`: Stores conversational turns (`user` and `assistant`), session IDs, timestamps, and owner `user_id`.
    - Creates indices on `user_id`, `session_id`, and `created_at` for high-performance lookups.
  - `reconcile_storage_layers()`: A self-healing startup procedure that identifies and fixes discrepancies between disk files, SQLite records, and FAISS vector chunks (cleans orphaned files, removes broken DB records, and purges dangling chunks).

#### [security.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/security.py)
- **Role**: Authentication and multi-tenant isolation.
- **What it does**:
  - `verify_api_key()`: FastAPI dependency that checks incoming `X-API-Key` headers against configured authorized keys.
  - Computes a deterministic SHA-256 hash of the API key to use as the caller's unique `user_id`.
  - Exposes context-variable `current_user_id` so downstream database queries and vector retrievals automatically filter data by the authenticated tenant.

#### [rate_limiter.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/rate_limiter.py)
- **Role**: Request rate limiting to prevent API abuse and cost overruns.
- **What it does**:
  - Implements an in-memory sliding-window token bucket algorithm keyed by the caller's `user_id`.
  - `check_chat_rate_limit()`: Enforces maximum queries per minute for chat endpoints.
  - `check_upload_rate_limit()`: Enforces maximum PDF uploads per minute for document ingestion.
  - Returns HTTP 429 Too Many Requests when limits are exceeded.

#### [backup.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/backup.py)
- **Role**: Automated backup creation and archive rotation.
- **What it does**:
  - `backup_data_assets()`: Copies the active SQLite database (using SQLite's online backup API to ensure snapshot consistency) and FAISS index into a compressed ZIP file in `data/backups/`.
  - Rotates backups by automatically deleting older archives when the count exceeds `BACKUP_RETENTION_COUNT` (default: 5).

#### [cache.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/core/cache.py)
- **Role**: In-memory query result cache with user-isolation and automatic invalidation.
- **What it does**:
  - `QueryCache`: Thread-safe, lock-guarded LRU cache with configurable time-to-live (`QUERY_CACHE_TTL_SECONDS`) and capacity (`QUERY_CACHE_MAX_SIZE`).
  - Keyed by `user_id:normalized_query` to prevent cross-tenant data leakage.
  - Automatically invalidated for the owning user whenever documents are uploaded or deleted.

---

### App API (`app/api/`)

#### [router.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/api/router.py)
- **Role**: Main API router.
- **What it does**: Mounts all feature-specific routers:
  - `/api/v1/chat` -> `endpoints.chat.router`
  - `/api/v1/documents` -> `endpoints.documents.router`
  - `/api/v1/sessions` -> `endpoints.sessions.router`
  - `/api/v1/vectorstore` -> `endpoints.vectorstore.router`

#### [deps.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/api/deps.py)
- **Role**: FastAPI dependency injection providers with concurrency safety.
- **What it does**:
  - `get_embeddings()`: Thread-safe singleton factory producing the active embedding model (`BAAI/bge-m3` via `HuggingFaceEmbeddings` or Google/OpenAI embeddings).
  - `get_llm()`: Factory for LangChain chat models (`ChatGoogleGenerativeAI`, `ChatOpenAI`, `ChatAnthropic`) with timeout and retry configuration. In `testing` mode, gracefully falls back to deterministic fakes.
  - `get_vector_store()`, `get_pdf_processor()`, `get_history_manager()`, and `get_rag_service()`: Provides instantiated, ready-to-use services to route handlers.

#### [endpoints/chat.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/api/endpoints/chat.py)
- **Role**: Chatbot query, token streaming, and evaluation route handlers.
- **What it does**:
  - `POST /api/v1/chat/query`: Main conversational endpoint. Loads chat history, executes the multi-stage RAG pipeline in a worker thread via `anyio.to_thread.run_sync`, saves new user and assistant turns to SQLite upon success, formats detailed citations, and returns performance timings. Checks query cache before retrieval.
  - `POST /api/v1/chat/stream`: Server-Sent Events (SSE) streaming endpoint. Yields initial source citations, progressive tokens as generated by the LLM, and completion metadata.
  - `POST /api/v1/chat/evaluate`: Computes quantitative retrieval and generation metrics (context precision, context recall, citation coverage, response latency) against reference ground truth for benchmarking.

#### [endpoints/documents.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/api/endpoints/documents.py)
- **Role**: Document management route handlers.
- **What it does**:
  - `POST /api/v1/documents/upload`: Validates file size (max 10MB) and PDF magic bytes (`%PDF-`), writes file to `data/uploads/`, extracts text chunks, inserts metadata records into SQLite, and indexes chunks into FAISS with UUID tracking.
  - `GET /api/v1/documents/list`: Returns a paginated list of uploaded documents for the authenticated user.
  - `GET /api/v1/documents/{document_id}`: Retrieves ingestion status and metadata for a specific document.
  - `DELETE /api/v1/documents/{document_id}`: Atomically removes document chunks from FAISS, deletes the PDF file from disk, and cascades deletion in SQLite.

#### [endpoints/sessions.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/api/endpoints/sessions.py)
- **Role**: Chat session history route handlers.
- **What it does**:
  - `GET /api/v1/sessions`: Lists all conversation sessions owned by the caller with message counts and last activity timestamps.
  - `GET /api/v1/sessions/{session_id}`: Returns all historical user and assistant messages for a session.
  - `DELETE /api/v1/sessions/{session_id}`: Clears message history for a session.

#### [endpoints/vectorstore.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/api/endpoints/vectorstore.py)
- **Role**: Diagnostic and vector store inspection endpoints.
- **What it does**:
  - `POST /api/v1/vectorstore/search`: Runs raw similarity search directly against FAISS without calling the LLM. Useful for evaluating embedding retrieval accuracy.
  - `GET /api/v1/vectorstore/status`: Returns vector index readiness, vector count, disk file size, and path (with absolute path masked for security).
  - `GET /api/v1/vectorstore/count`: Lightweight endpoint returning total vector count.

---

### App Services (`app/services/`)

#### [pdf_processor.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/services/pdf_processor.py)
- **Role**: PDF parsing, text extraction, and chunking.
- **What it does**:
  - Uses `fitz` (PyMuPDF) to extract clean text on a per-page basis.
  - Preserves critical source metadata: `filename`, `page_number`, and `total_pages`.
  - Splits text into chunks using `RecursiveCharacterTextSplitter` (default 1,000 characters with 200-character overlap) to preserve semantic continuity across chunk boundaries.

#### [rag_service.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/services/rag_service.py)
- **Role**: The core RAG pipeline orchestrator.
- **What it does**:
  1. **Query Caching**: Checks the in-memory LRU TTL query cache before retrieval; returns immediate cached results on repeated queries.
  2. **Query Reformulation**: If conversation history exists, uses a light LLM call to rewrite ambiguous follow-up questions (e.g., "What are its types?") into standalone search queries.
  3. **Dense Retrieval**: Queries FAISS for top-K candidate chunks using cosine similarity, filtered by user ID.
  4. **Adaptive Cross-Encoder Reranking**: If enabled, passes query-document pairs to the cross-encoder, skipping or throttling rerank steps when top dense similarity exceeds the adaptive threshold.
  5. **Relevance Gating ("I don't know")**: Compares top candidate scores against `RELEVANCE_THRESHOLD`. If no chunks meet the threshold, the system immediately returns a helpful "I do not have sufficient information in the uploaded documents" message, skipping the LLM generation call to eliminate hallucinations.
  6. **Grounded Generation & Polymorphic Content Normalization**: Formats system prompts with numbered inline sources `[Source X]`. Uses `_extract_text_content` to safely normalize string, list-of-string, or structured dict block responses from providers like Google Gemini and Anthropic.
  7. **Real-Time Streaming**: Implements `answer_query_stream()` to yield Server-Sent Events (SSE) token chunks for fast time-to-first-token.
  8. **Citation Resolution**: Parses returned text for citations and links them to exact document names, page numbers, and text snippets.

#### [reranker.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/services/reranker.py)
- **Role**: Neural cross-encoder reranking.
- **What it does**:
  - Loads a Hugging Face `CrossEncoder` model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) lazily on first use.
  - Uses a process-wide thread lock to serialize CPU inference and prevent memory spikes.
  - Transforms raw model output logits through a sigmoid function `1 / (1 + exp(-x))` to provide normalized 0.0 to 1.0 confidence scores.

#### [history_manager.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/services/history_manager.py)
- **Role**: Persistent conversational turn manager.
- **What it does**:
  - Reads and writes chat turns to the `chat_history` SQLite table.
  - Offloads database queries to worker threads to avoid blocking FastAPI's async event loop.
  - Ensures queries are scoped by `user_id` to prevent session leaking across users.

#### [errors.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/services/errors.py)
- **Role**: Provider error classification and safe HTTP mapping.
- **What it does**:
  - Inspects upstream LLM exceptions (timeouts, quota exceeded, rejected keys, internal provider 500s) and categorizes them into explicit classes (`ProviderTimeoutError`, `ProviderQuotaError`, `ProviderAuthError`, `GenerationError`).
  - Maps errors to appropriate HTTP status codes (504 Gateway Timeout, 503 Service Unavailable, 502 Bad Gateway) with actionable user messages while preventing internal credential or stack trace leakage to the client.

---

### App Vector Store (`app/vectorstore/`)

#### [native_faiss.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/vectorstore/native_faiss.py)
- **Role**: Native, pickle-free FAISS index adapter.
- **What it does**:
  - Replaces LangChain's deprecated `langchain_community.vectorstores.FAISS` implementation.
  - Uses native `faiss.IndexFlatIP` with L2-normalized vectors to compute exact cosine similarity.
  - Stores document content and metadata in human-readable JSON files (`manifest.json`), eliminating Python `pickle` deserialization security vulnerabilities.
  - Supports vector removal by ID (`remove_ids`), enabling true document deletion without rebuilding the entire index.
  - Enforces embedding dimension matching to detect index compatibility issues.

#### [faiss_store.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/vectorstore/faiss_store.py)
- **Role**: High-level FAISS vector store manager.
- **What it does**:
  - Wraps `native_faiss.FAISS` with thread safety using a module-level `threading.RLock()` (`_faiss_write_lock`) for all write operations (`add_documents`, `delete_documents`, `save_index`).
  - Performs multi-tenant filtering in `similarity_search` so callers only retrieve chunks matching their authenticated `user_id`.
  - Detects legacy index formats and informs operators to run the reindexing tool.

---

### App Prompts & Schemas (`app/prompts/`, `app/schemas/`)

#### [prompts/templates.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/prompts/templates.py)
- **Role**: Prompt definitions and guidelines.
- **What it does**:
  - `CONDENSE_QUESTION_PROMPT`: Instructs the LLM to rewrite a conversational follow-up into a standalone search query.
  - `QA_SYSTEM_PROMPT`: Rigorous system instructions for RAG answering:
    - Answer strictly using provided context.
    - If context does not contain the answer, state that information is not available.
    - Cite sources inline using `[Source X]` bracket notations.

#### [schemas/chat.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/schemas/chat.py)
- **Role**: Data contracts for chat operations.
- **What it does**:
  - Defines `ChatQuery` with validation (1-4000 character constraints, whitespace stripping).
  - Defines `ChatResponse`, `SourceDocumentSchema`, `PipelineTimings`, `ChatResponseMetadata`, `SessionListResponse`, and `EvaluationResponse`.

#### [schemas/document.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/schemas/document.py)
- **Role**: Data contracts for document operations.
- **What it does**:
  - Defines `DocumentUploadResponse`, `DocumentStatusResponse`, `SearchQuerySchema`, and `SearchResultDocument`.

---

### App Frontend (`app/static/`)

#### [index.html](file:///c:/PROJECTS/rag-pdf-chatbot/app/static/index.html), [styles.css](file:///c:/PROJECTS/rag-pdf-chatbot/app/static/styles.css), [app.js](file:///c:/PROJECTS/rag-pdf-chatbot/app/static/app.js)
- **Role**: Built-in responsive Web UI (Atlas PDF Intelligence Workspace).
- **What it does**:
  - Provides a single-page chat workspace served directly by FastAPI.
  - Allows uploading PDF files via drag-and-drop or file picker with live status feedback.
  - Displays conversation messages, citation badges, source library card, and retrieval status.
  - Includes session history sidebar with new chat creation, session switching, and theme toggling.

---

### App Entrypoint (`app/main.py`)

#### [main.py](file:///c:/PROJECTS/rag-pdf-chatbot/app/main.py)
- **Role**: FastAPI application factory and lifecycle orchestrator.
- **What it does**:
  - `lifespan`: Ensures required directories (`data/uploads`, `data/faiss_index`, `data/backups`) exist, initializes SQLite schema, executes self-healing storage reconciliation, and triggers startup snapshot backups.
  - Security headers middleware: Injects strict CSP, `X-Content-Type-Options: nosniff`, and referrer policies.
  - Custom documentation routes: Serves `/docs`, `/redoc`, and `/docs/oauth2-redirect` using dynamic cryptographic nonces to support Swagger CDN assets safely without compromising main application CSP.
  - Health check endpoints:
    - `GET /health`: Liveness probe (200 OK).
    - `GET /health/ready`: Readiness probe verifying SQLite database connectivity and FAISS index existence (returns 503 if dependencies fail).

---

### Operational Scripts (`scripts/`)

#### [reindex.py](file:///c:/PROJECTS/rag-pdf-chatbot/scripts/reindex.py)
- **Role**: Offline / maintenance index regeneration CLI.
- **What it does**:
  - Iterates over all completed documents in SQLite.
  - Re-reads source PDFs from `data/uploads/`.
  - Re-extracts and chunks text with the active text splitter settings.
  - Generates new embeddings using the currently active embedding model and rebuilds the FAISS index cleanly from scratch.

---

### Automated Tests (`tests/`)

The test suite consists of **95 comprehensive pytest tests**:
- [test_api.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_api.py): Basic route status, authorization checks, and simple health endpoints.
- [test_arch_problems.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_arch_problems.py): Concurrency locking, WAL configuration, PDF validation, chunk tracking, and document deletion.
- [test_chat_failures.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_chat_failures.py): Provider timeouts (504), quota limits (503), rejected credentials (502), and ensuring failed generations do not corrupt chat history.
- [test_dependency_cache_and_docs.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_dependency_cache_and_docs.py): Singleton thread-safe embedding instantiation and Swagger/ReDoc CSP nonce verification.
- [test_e2e.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_e2e.py): End-to-end PDF upload and multi-turn chat with source citations.
- [test_missing_apis.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_missing_apis.py): Paginated session listing, session deletion, document details, and readiness probes.
- [test_multi_tenant.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_multi_tenant.py): Multi-tenant isolation (user documents, chat history, and vector retrieval boundaries).
- [test_operational.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_operational.py): Rate limiting, input constraints, automated backups, and file secret resolution.
- [test_p2_polish.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_p2_polish.py): Reranking integration, relevance gating ("I don't know"), and path masking.
- [test_reconciliation.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_reconciliation.py): Storage reconciliation fixing orphaned disk files, orphaned DB records, and dangling vector chunks.
- [test_services.py](file:///c:/PROJECTS/rag-pdf-chatbot/tests/test_services.py): Unit tests for PDF processor, history manager, FAISS vector store, and provider factories.
