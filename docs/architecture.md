# RAG Chatbot System Architecture

This document describes the flow and architecture of the Retrieval-Augmented Generation (RAG) system.

## Ingestion Pipeline Flow

```mermaid
graph TD
    A[User PDF Upload] --> B[FastAPI Endpoint]
    B --> C[PDF Processor Service]
    C --> D[Text Extraction]
    D --> E[Text Splitter / Chunking]
    E --> F[Generate Embeddings]
    F --> G[Update FAISS Index]
    G --> H[Save FAISS Index to disk]
```

## Retrieval and QA Flow

```mermaid
graph TD
    1[User Question] --> 2[FastAPI Chat Endpoint]
    2 --> 3[RAG Service]
    3 --> 4[Query Embeddings Generator]
    4 --> 5[FAISS Similarity Search]
    5 --> 6[Retrieve Context Documents]
    6 --> 7[Format QA Chat Prompt]
    7 --> 8[Invoke LLM Provider]
    8 --> 9[Return Structured Answer]
```

## Modular Layers

1. **API Layer (`app/api`)**: Receives requests, handles authentication, returns Pydantic DTO responses.
2. **Services Layer (`app/services`)**: Business logic (parsing documents, configuring LangChain chains, calling APIs).
3. **Vector Store Layer (`app/vectorstore`)**: Database interface wrapper wrapping FAISS index interactions.
4. **Schemas Layer (`app/schemas`)**: Validates input payloads and structures JSON outputs.
5. **Prompts Layer (`app/prompts`)**: Manages model templates.
6. **Core Layer (`app/core`)**: Handles security dependencies and global settings loading.
