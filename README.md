# FastAPI RAG Chatbot using LangChain and FAISS

A production-ready FastAPI codebase structure for a Retrieval-Augmented Generation (RAG) chatbot using LangChain and a FAISS local vector store. This project features a clean, modular layout separating endpoints, schemas, vector storage wrapper classes, prompt templates, and PDF ingestion services.

## Project Structure

```text
rag-pdf-chatbot/
├── data/                  # Local directory for raw documents and vectorstore indexes
├── docs/                  # Project documentation and architectural charts
├── app/                   # Source code
│   ├── __init__.py
│   ├── main.py            # FastAPI app configuration & entrypoint
│   ├── api/               # API Router & Route handlers
│   │   ├── __init__.py
│   │   ├── deps.py        # Dependency Injection helpers (DB, settings, services)
│   │   ├── router.py      # Combines API routes
│   │   └── endpoints/     # Feature-specific route files
│   │       ├── __init__.py
│   │       ├── chat.py    # Main chatbot / conversation endpoints
│   │       ├── documents.py # Document ingestion / upload endpoints
│   │       └── vectorstore.py # Direct vectorstore queries & index status
│   ├── core/              # Global configurations, credentials, security & exceptions
│   │   ├── __init__.py
│   │   ├── config.py      # Pydantic Settings implementation
│   │   └── security.py    # API Key validator / Authentication middleware
│   ├── prompts/           # LLM Chat & System Prompts
│   │   ├── __init__.py
│   │   └── templates.py   # LangChain prompt templates
│   ├── schemas/           # Pydantic models (data validation & DTOs)
│   │   ├── __init__.py
│   │   ├── chat.py        # Request/Response schemas for chat conversations
│   │   └── document.py    # Upload status / Metadata schemas
│   ├── services/          # Business logic wrappers / Services
│   │   ├── __init__.py
│   │   ├── pdf_processor.py # Extracting & chunking PDFs
│   │   └── rag_service.py # Orchestrating retrieval and response generation
│   ├── utils/             # Reusable helper utilities
│   │   ├── __init__.py
│   │   └── helpers.py     # Logging setup & general helpers
│   └── vectorstore/       # Vector database integration
│       ├── __init__.py
│       └── faiss_store.py # FAISS load/save/query operations
├── tests/                 # Unit & Integration Tests
│   ├── __init__.py
│   ├── conftest.py        # Test configuration & fixtures
│   ├── test_api.py        # Route-level endpoint tests
│   └── test_services.py   # Unit tests for services
├── .env.example           # Reference configuration variables
├── .gitignore             # Standard git exclusions
└── requirements.txt       # Project python packages
```

## Setup & Running

1. **Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Copy `.env.example` to `.env` and fill in API keys:
   ```bash
   cp .env.example .env
   ```

3. **Running the Server**:
   ```bash
   uvicorn app.main:app --reload
   ```
   Access the interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

4. **Running Tests**:
   ```bash
   pytest
   ```
