"""Lazy, process-wide cross-encoder; scores are sigmoid-transformed logits."""
import threading
import numpy as np
from app.core.config import settings

_model = None
_model_name = None
_lock = threading.Lock()


def rerank(query, documents):
    global _model, _model_name
    if not documents:
        return []
    # Serialize CPU inference and initialization to keep memory usage bounded.
    with _lock:
        if _model is None or _model_name != settings.RERANK_MODEL_NAME:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(settings.RERANK_MODEL_NAME, device=settings.EMBEDDING_DEVICE, max_length=512)
            _model_name = settings.RERANK_MODEL_NAME
        import torch
        logits = np.asarray(_model.predict(
            [(query, doc.page_content) for doc in documents],
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            activation_fn=torch.nn.Identity(), show_progress_bar=False,
        ), dtype=float).reshape(-1)
    if len(logits) != len(documents) or not np.isfinite(logits).all():
        raise ValueError("Reranker returned invalid scores")
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    ranked = []
    for doc, score in zip(documents, scores):
        copy = doc.model_copy(deep=True)
        copy.metadata["rerank_score"] = float(score)
        ranked.append(copy)
    return sorted(ranked, key=lambda doc: doc.metadata["rerank_score"], reverse=True)
