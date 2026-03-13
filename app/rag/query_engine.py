"""Query helpers for the financial RAG knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex

from app.rag.index_builder import DEFAULT_INDEX_DIR, load_financial_index

logger = logging.getLogger(__name__)


def query_financial_docs(
    query_text: str,
    index: VectorStoreIndex | None = None,
    persist_dir: str | Path = DEFAULT_INDEX_DIR,
    top_k: int = 3,
) -> dict[str, Any]:
    """Retrieve relevant snippets from the financial knowledge base.

    Args:
        query_text: Natural-language question to search for.
        index: Optional pre-loaded index instance.
        persist_dir: Persisted index path used when ``index`` is not supplied.
        top_k: Maximum number of snippets to return.

    Returns:
        A dictionary with the query and ranked text snippets.

    Raises:
        ValueError: For invalid query/top_k values or empty retrieval results.
        RuntimeError: If retrieval fails due to engine errors.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        logger.error("Empty or invalid query text received")
        raise ValueError("query_text must be a non-empty string")

    if top_k <= 0:
        logger.error("Invalid top_k value: %s", top_k)
        raise ValueError("top_k must be greater than 0")

    query = query_text.strip()
    rag_index = index if index is not None else load_financial_index(persist_dir)

    try:
        retriever = rag_index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
    except Exception as exc:
        logger.exception("Financial document retrieval failed for query=%r", query)
        raise RuntimeError("Failed to query financial documents") from exc

    if not nodes:
        logger.warning("No retrieval results found for query=%r", query)
        raise ValueError("No relevant snippets found for the query")

    snippets: list[dict[str, Any]] = []
    for node in nodes:
        content = node.get_content().strip()
        if not content:
            continue
        snippets.append(
            {
                "text": content,
                "score": float(getattr(node, "score", 0.0) or 0.0),
            }
        )

    if not snippets:
        logger.warning("Retriever returned nodes but no usable text for query=%r", query)
        raise ValueError("No relevant snippets found for the query")

    logger.info("Retrieved %d snippets for query=%r", len(snippets), query)
    return {"query": query, "snippets": snippets}
