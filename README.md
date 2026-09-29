# RAG PDF Chatbot

A fast, production-ready Retrieval-Augmented Generation (RAG) chatbot backend and web application built with **FastAPI**, **FAISS**, **BAAI/bge-m3 embeddings**, and **Google Gemini (or OpenAI / Anthropic)**. 

Upload PDF documents, ask questions, and receive grounded answers complete with inline source and page citations, persistent chat history, and cross-encoder reranking.

---

## Features

- **Document Processing**: Extracts text and per-page metadata from uploaded PDFs with smart chunking (1,000 characters, 200 overlap) and SHA-256 chunk deduplication.
- **Dense Vector Search**: Native FAISS index with inner-product cosine similarity, unit-normalized embeddings, optional HNSW graph indexing, and MMAP memory-mapped snapshot loading.
- **Precision Reranking**: Cross-encoder reranking with sigmoid confidence scores, adaptive rerank gating to skip CPU load on clear matches, and an "I don't know" threshold to filter hallucinations.
- **Server-Sent Events (SSE) Streaming**: Real-time token-by-token streaming at `/api/v1/chat/stream` with live citation previews in the Web UI.
- **In-Memory Query Result Cache**: User-isolated LRU query cache with TTL expiration and automatic invalidation on document mutations.
- **Persistent Storage**: SQLite database for chat history, sessions, document tracking, and chunk mappings.
- **Security & Multi-Tenancy**: API key authentication with SHA-256 key hashing for per-user data isolation and token-bucket rate limiting.
- **Clean Web UI & API**: Modern browser interface with live token streaming served directly from FastAPI, plus interactive Swagger UI at `/docs`.

---

## Documentation

Comprehensive guides are available in the [`docs/`](docs/) directory:

- [Codebase Structure & File Catalog](docs/codebase-structure-and-guide.md) - Explains every folder, file, and code component.
- [Problems Faced & Solutions](docs/problems-and-solutions.md) - Deep dive into issues encountered, solutions, and architectural trade-offs.
- [Step-by-Step Technical Build Guide](docs/technical-build-guide.md) - Technical walkthrough of frameworks, databases, embeddings, and LLM APIs.
- [System Architecture](docs/architecture.md) - Visual diagrams of ingestion, retrieval, and generation flows.

---

## Project Structure

```text
rag-pdf-chatbot/
├── app/
│   ├── api/               # API routers, endpoints (chat, documents, sessions, vectorstore) & dependencies
│   ├── core/              # Configuration, SQLite database, security, rate limiting, and backup
│   ├── prompts/           # System prompts and conversational templates
│   ├── schemas/           # Pydantic data models for requests, responses, and metrics
│   ├── services/          # PDF processor, RAG pipeline, cross-encoder reranker, and history manager
│   ├── static/            # Frontend web UI (HTML, CSS, Vanilla JS)
│   ├── vectorstore/       # Native FAISS vector store adapter and store wrapper
│   └── main.py            # FastAPI application entrypoint & lifecycle management
├── data/                  # SQLite database, uploaded PDFs, and FAISS index files
├── docs/                  # In-depth architectural, technical, and troubleshooting documentation
├── scripts/               # Utility scripts (e.g., reindexing script)
├── tests/                 # Comprehensive automated pytest test suite (110 tests)
├── .env.example           # Environment variable template
├── Dockerfile             # Multi-stage production container build
├── requirements.txt       # Python dependencies
└── requirements.lock      # Pinned dependency lockfile
```

---

## Quick Start

### 1. Set Up Python Environment

Using Python 3.12 or 3.13:

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment configuration:

```bash
cp .env.example .env
```

Open `.env` and set your API keys:
- `API_KEY`: Set your secret application API key (e.g. `my-secure-key`).
- `GOOGLE_API_KEY`: Your Gemini API key.
- `LLM_PROVIDER`: `google` (or `openai`, `anthropic`).
- `LLM_MODEL_NAME`: `gemma-4-26b-a4b-it` or `gemini-1.5-flash`.

### 3. Run the Application

```bash
uvicorn app.main:app --reload
```

- **Web UI**: Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
- **Interactive API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Running Tests

Execute the full automated test suite (ensure `.venv` is activated, or use its direct path):

```bash
# With activated virtual environment (.venv)
pytest

# Or directly without activating:
.\.venv\Scripts\pytest        # On Windows
# .venv/bin/pytest            # On Linux/macOS
```

All **110 tests** validate retrieval, reranking, multi-tenancy, rate limiting, persistence, security headers, CORS, streaming, and error handling.

---

## Production Deployment

### Option 1: Docker Compose (Recommended)

1. Configure your production environment variables in `.env` (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```
   Set strong values for:
   - `API_KEY`: A strong secret key.
   - `ALLOWED_ORIGINS`: Comma-separated allowed domains (e.g. `https://yourdomain.com`).
   - `GOOGLE_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).
   - `APP_ENV=production`

2. Launch the containerized service:
   ```bash
   docker compose up -d --build
   ```

3. View running logs:
   ```bash
   docker compose logs -f
   ```

The container automatically runs as an unprivileged user (`appuser`), checks container health at `/health`, and persists database assets and uploads in `./data`.

### Option 2: Standalone Docker

```bash
docker build -t rag-pdf-chatbot:latest .
docker run -d \
  --name rag-pdf-chatbot \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  rag-pdf-chatbot:latest
```

### Option 3: Production ASGI Server

Run Uvicorn with production settings behind a reverse proxy (Nginx, Caddy, or Traefik) providing SSL/TLS termination:

```bash
python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --proxy-headers \
  --forwarded-allow-ips "*"
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
