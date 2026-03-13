"""Build and load the FAISS-backed financial RAG index."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import MockEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md")
DEFAULT_DOCUMENTS_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "financial_docs"
DEFAULT_INDEX_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "faiss_index"
DEFAULT_EMBED_DIM: int = 384


def build_financial_index(
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    persist_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_dim: int = DEFAULT_EMBED_DIM,
) -> VectorStoreIndex:
    """Build a FAISS vector index from financial documents.

    Args:
        documents_dir: Directory containing source documents.
        persist_dir: Directory where index artifacts are persisted.
        embedding_dim: Embedding dimension used by the embedding model.

    Returns:
        The built ``VectorStoreIndex`` instance.

    Raises:
        FileNotFoundError: If the source documents directory does not exist.
        NotADirectoryError: If ``documents_dir`` is not a directory.
        ValueError: If no supported documents are found.
        RuntimeError: If embedding/index construction fails.
    """
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        logger.error("Documents directory not found: %s", docs_path)
        raise FileNotFoundError(f"Documents directory not found: {docs_path}")

    if not docs_path.is_dir():
        logger.error("Documents path is not a directory: %s", docs_path)
        raise NotADirectoryError(f"Expected a directory path: {docs_path}")

    input_files = [
        file_path
        for file_path in docs_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not input_files:
        logger.error("No supported documents found in %s", docs_path)
        raise ValueError("No financial documents found to index")

    reader = SimpleDirectoryReader(input_files=[str(path) for path in input_files])
    documents = reader.load_data()
    if not documents:
        logger.error("Document loader returned no documents for %s", docs_path)
        raise ValueError("No readable documents found to index")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    faiss_index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = MockEmbedding(embed_dim=embedding_dim)

    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    except Exception as exc:
        logger.exception("Failed to build vector index due to embedding/index error")
        raise RuntimeError("Failed to build financial RAG index") from exc

    storage_context.persist(persist_dir=str(persist_path))
    logger.info(
        "Built FAISS financial index: docs=%d persist_dir=%s",
        len(documents),
        persist_path,
    )
    return index


def load_financial_index(
    persist_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_dim: int = DEFAULT_EMBED_DIM,
) -> VectorStoreIndex:
    """Load a persisted FAISS-backed ``VectorStoreIndex``.

    Args:
        persist_dir: Directory containing persisted index files.
        embedding_dim: Embedding dimension used at index build time.

    Returns:
        Loaded ``VectorStoreIndex`` instance.

    Raises:
        FileNotFoundError: If ``persist_dir`` does not exist.
        ValueError: If ``persist_dir`` exists but is empty.
        RuntimeError: If index loading fails.
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        logger.error("Persist directory not found: %s", persist_path)
        raise FileNotFoundError(f"Persist directory not found: {persist_path}")

    if not any(persist_path.iterdir()):
        logger.error("Persist directory is empty: %s", persist_path)
        raise ValueError("Persist directory is empty; build the index first")

    try:
        vector_store = FaissVectorStore.from_persist_dir(str(persist_path))
        storage_context = StorageContext.from_defaults(
            persist_dir=str(persist_path),
            vector_store=vector_store,
        )
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=MockEmbedding(embed_dim=embedding_dim),
        )
    except Exception as exc:
        logger.exception("Failed to load persisted FAISS index from %s", persist_path)
        raise RuntimeError("Failed to load financial RAG index") from exc

    logger.info("Loaded FAISS financial index from %s", persist_path)
    return index
