"""Small application-owned FAISS adapter; JSON metadata, no pickle execution.

Vectors are unit-normalized and searched by inner product (cosine similarity).
Each save writes an immutable vector snapshot and atomically publishes a manifest.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class IndexCompatibilityError(RuntimeError):
    pass


def embedding_identity(embeddings):
    model = getattr(embeddings, "model_name", None) or getattr(embeddings, "model", None)
    if not isinstance(model, str):
        model = f"{type(embeddings).__module__}.{type(embeddings).__name__}:{getattr(embeddings, 'size', '')}"
    return model


class DocumentStore:
    def __init__(self, documents=None):
        self._dict = documents or {}

    def search(self, document_id):
        return self._dict.get(document_id)


class FAISS(VectorStore):
    def __init__(self, embeddings, index, documents=None, ids=None):
        self._embeddings = embeddings
        self.index = index
        self.docstore = DocumentStore(documents)
        self.index_to_docstore_id = dict(enumerate(ids or []))

    @property
    def embeddings(self):
        return self._embeddings

    @staticmethod
    def _vectors(values, dimension=None):
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Embeddings must be a finite two-dimensional matrix")
        if dimension is not None and matrix.shape[1] != dimension:
            raise IndexCompatibilityError("Embedding dimension changed. Rebuild the index from the PDFs.")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("Embedding provider returned a zero vector")
        matrix = np.ascontiguousarray(matrix)
        faiss.normalize_L2(matrix)
        return matrix

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, ids=None, **kwargs):
        docs = [Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas or [{} for _ in texts])]
        return cls.from_documents(docs, embedding, ids=ids)

    @classmethod
    def from_documents(cls, documents, embedding, ids=None, **kwargs):
        if not documents:
            raise ValueError("At least one document is required to determine vector dimension")
        vectors = cls._vectors(embedding.embed_documents([doc.page_content for doc in documents]))
        from app.core.config import settings
        if getattr(settings, "FAISS_INDEX_TYPE", "flat") == "hnsw":
            raw_index = faiss.IndexHNSWFlat(vectors.shape[1], settings.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
            raw_index.hnsw.efSearch = settings.FAISS_HNSW_EF_SEARCH
        else:
            raw_index = faiss.IndexFlatIP(vectors.shape[1])
        instance = cls(embedding, raw_index)
        instance._add(documents, ids, vectors)
        return instance

    def _add(self, documents, ids, vectors):
        ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in documents]
        if len(ids) != len(documents) or len(vectors) != len(documents):
            raise ValueError("Documents, vectors, and IDs must have the same length")
        if len(set(ids)) != len(ids) or set(ids) & self.docstore._dict.keys():
            raise ValueError("Chunk IDs must be unique")
        offset = self.index.ntotal
        self.index.add(vectors)
        for i, (doc_id, doc) in enumerate(zip(ids, documents)):
            self.docstore._dict[doc_id] = doc.model_copy(deep=True)
            self.index_to_docstore_id[offset + i] = doc_id
        return ids

    def add_documents(self, documents, ids=None, **kwargs):
        vectors = self._vectors(self.embeddings.embed_documents([d.page_content for d in documents]), self.index.d)
        return self._add(documents, ids, vectors)

    def delete(self, ids, **kwargs):
        missing = set(ids) - self.docstore._dict.keys()
        if missing:
            raise ValueError(f"Unknown chunk IDs: {sorted(missing)}")
        positions = np.asarray([i for i, doc_id in self.index_to_docstore_id.items() if doc_id in ids], dtype=np.int64)
        self.index.remove_ids(positions)
        remaining = [doc_id for doc_id in self.index_to_docstore_id.values() if doc_id not in ids]
        self.index_to_docstore_id = dict(enumerate(remaining))
        for doc_id in ids:
            del self.docstore._dict[doc_id]
        return True

    def similarity_search_with_score(self, query, k=4, filter=None, fetch_k=None, **kwargs):
        if k <= 0 or self.index.ntotal == 0:
            return []
        vector = self._vectors([self.embeddings.embed_query(query)], self.index.d)
        # Scan all candidates before tenant filtering so other users cannot crowd
        # a tenant's documents out of the candidate pool.
        scores, positions = self.index.search(vector, self.index.ntotal if filter else min(k, self.index.ntotal))
        result = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            doc_id = self.index_to_docstore_id[int(position)]
            doc = self.docstore.search(doc_id)
            accepted = filter(doc.metadata) if callable(filter) else all(doc.metadata.get(key) == value for key, value in (filter or {}).items())
            if not accepted:
                continue
            copy = doc.model_copy(deep=True)
            copy.metadata.setdefault("chunk_id", doc_id)
            result.append((copy, float(np.clip(score, -1, 1))))
            if len(result) == k:
                break
        return result

    def similarity_search(self, query, k=4, **kwargs):
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def save_local(self, folder_path):
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"index-{uuid.uuid4().hex}.faiss"
        vector_path = folder / name
        faiss.write_index(self.index, str(vector_path))
        payload = {
            "version": 1, "metric": "cosine", "dimension": self.index.d,
            "embedding_model": embedding_identity(self.embeddings), "vector_file": name,
            "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
            "documents": [{"id": doc_id, "text": self.docstore.search(doc_id).page_content,
                           "metadata": self.docstore.search(doc_id).metadata}
                          for doc_id in self.index_to_docstore_id.values()],
        }
        temporary = folder / f"manifest-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, folder / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load_local(cls, folder_path, embeddings, **kwargs):
        folder = Path(folder_path)
        manifest = folder / "manifest.json"
        if not manifest.exists():
            raise IndexCompatibilityError("Legacy FAISS index detected. Run python -m scripts.reindex before starting the server.")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("version") != 1 or data.get("metric") != "cosine":
            raise IndexCompatibilityError("Unsupported vector index format; rebuild the index.")
        if data.get("embedding_model") != embedding_identity(embeddings):
            raise IndexCompatibilityError("Embedding model differs from the saved index. Run python -m scripts.reindex.")
        name = data["vector_file"]
        if Path(name).name != name or not name.endswith(".faiss"):
            raise ValueError("Invalid vector snapshot path")
        vector_path = folder / name
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != data["vector_sha256"]:
            raise ValueError("Vector snapshot checksum mismatch; restore a backup")
        from app.core.config import settings
        if getattr(settings, "FAISS_MMAP_ENABLED", False):
            index = faiss.read_index(str(vector_path), faiss.IO_FLAG_MMAP)
        else:
            index = faiss.read_index(str(vector_path))
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = getattr(settings, "FAISS_HNSW_EF_SEARCH", 64)
        rows = data["documents"]
        if index.d != data["dimension"] or index.ntotal != len(rows) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Vector snapshot and document manifest disagree")
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs in manifest")
        docs = {row["id"]: Document(page_content=row["text"], metadata=row["metadata"]) for row in rows}
        return cls(embeddings, index, docs, ids)
