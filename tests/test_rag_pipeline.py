"""Unit tests for FAISS-backed RAG index build and query."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_index.core.embeddings import MockEmbedding

from app.rag.index_builder import build_financial_index, load_financial_index
from app.rag.query_engine import query_financial_docs


class _EmptyRetriever:
    """Test helper retriever that simulates empty search results."""

    def retrieve(self, _: str) -> list[object]:
        return []


class _EmptyIndex:
    """Test helper index that returns an empty retriever."""

    def as_retriever(self, similarity_top_k: int = 3) -> _EmptyRetriever:  # noqa: ARG002
        return _EmptyRetriever()


class TestRagPipeline(unittest.TestCase):
    """Validate RAG build/load/query behavior and error handling."""

    def test_build_financial_index_missing_documents_dir(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_financial_index(documents_dir="this/path/does/not/exist")

    def test_build_financial_index_with_no_supported_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "ignored.json").write_text('{"x": 1}', encoding="utf-8")

            with self.assertRaises(ValueError):
                build_financial_index(documents_dir=docs_dir)

    def test_build_financial_index_handles_embedding_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "finance.txt").write_text("P/E ratio compares price to earnings.", encoding="utf-8")

            with patch("app.rag.index_builder._get_embedding_model", return_value=MockEmbedding(embed_dim=384)):
                with patch("app.rag.index_builder.VectorStoreIndex.from_documents", side_effect=Exception("embed fail")):
                    with self.assertRaises(RuntimeError):
                        build_financial_index(documents_dir=docs_dir)

    def test_query_financial_docs_returns_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            index_dir = Path(temp_dir) / "index"
            docs_dir.mkdir(parents=True, exist_ok=True)

            (docs_dir / "pe_ratio.txt").write_text(
                "P/E ratio means price-to-earnings ratio and is used for valuation.",
                encoding="utf-8",
            )
            (docs_dir / "market_cap.txt").write_text(
                "Market capitalization is share price multiplied by outstanding shares.",
                encoding="utf-8",
            )

            with patch("app.rag.index_builder._get_embedding_model", return_value=MockEmbedding(embed_dim=384)):
                build_financial_index(documents_dir=docs_dir, persist_dir=index_dir)
                loaded_index = load_financial_index(persist_dir=index_dir)
                result = query_financial_docs("What is P/E ratio?", index=loaded_index, top_k=2)

            self.assertIn("snippets", result)
            self.assertGreater(len(result["snippets"]), 0)
            snippet_texts = [snippet["text"] for snippet in result["snippets"]]
            self.assertTrue(any("P/E ratio" in text for text in snippet_texts))

    def test_query_financial_docs_handles_empty_results(self) -> None:
        with self.assertRaises(ValueError):
            query_financial_docs("Any query", index=_EmptyIndex())


if __name__ == "__main__":
    unittest.main()
