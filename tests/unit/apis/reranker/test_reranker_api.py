# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.common.responses import Order
from llama_stack.apis.models import Model
from llama_stack.apis.reranker import (
    ListModelsResponse,
    Reranker,
    RerankResponse,
    RerankResult,
)


class TestRerankerAPI:
    """Unit tests for the Reranker API protocol."""

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock reranker instance."""
        reranker = MagicMock(spec=Reranker)
        reranker.rerank = AsyncMock()
        reranker.list_models = AsyncMock()
        return reranker

    async def test_rerank_basic(self, mock_reranker):
        """Test basic reranking functionality."""
        # Setup mock response
        expected_results = [
            RerankResult(index=1, relevance_score=0.95),
            RerankResult(index=0, relevance_score=0.85),
            RerankResult(index=2, relevance_score=0.75),
        ]
        mock_reranker.rerank.return_value = RerankResponse(
            results=expected_results,
            model="rerank-test-model",
        )

        # Call rerank
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of AI.",
            "ML involves training models on data.",
            "Deep learning uses neural networks.",
        ]

        response = await mock_reranker.rerank(
            query=query,
            documents=documents,
            model="rerank-test-model",
        )

        # Verify the call
        mock_reranker.rerank.assert_called_once_with(
            query=query,
            documents=documents,
            model="rerank-test-model",
        )

        # Check response
        assert response.model == "rerank-test-model"
        assert len(response.results) == 3
        assert response.results[0].index == 1
        assert response.results[0].relevance_score == 0.95

    async def test_rerank_with_options(self, mock_reranker):
        """Test reranking with optional parameters."""
        # Setup mock response with documents
        expected_results = [
            RerankResult(
                index=1,
                relevance_score=0.95,
                document="Document B content",
            ),
            RerankResult(
                index=0,
                relevance_score=0.85,
                document="Document A content",
            ),
        ]
        mock_reranker.rerank.return_value = RerankResponse(
            results=expected_results,
            model="rerank-test-model",
            usage={"total_tokens": 150},
        )

        # Call rerank with options
        query = "test query"
        documents = ["Document A content", "Document B content", "Document C content"]

        response = await mock_reranker.rerank(
            query=query,
            documents=documents,
            model="rerank-test-model",
            top_n=2,
            truncation=True,
            return_documents=True,
        )

        # Verify the call
        mock_reranker.rerank.assert_called_once_with(
            query=query,
            documents=documents,
            model="rerank-test-model",
            top_n=2,
            truncation=True,
            return_documents=True,
        )

        # Check response
        assert len(response.results) == 2
        assert response.results[0].document == "Document B content"
        assert response.usage == {"total_tokens": 150}

    async def test_list_models(self, mock_reranker):
        """Test listing available reranker models."""
        # Setup mock response
        expected_models = [
            Model(
                identifier="rerank-model-1",
                provider_id="test-provider",
                metadata={"max_documents": 100},
            ),
            Model(
                identifier="rerank-model-2",
                provider_id="test-provider",
                metadata={"max_documents": 1000},
            ),
        ]
        mock_reranker.list_models.return_value = ListModelsResponse(models=expected_models)

        # Call list_models
        response = await mock_reranker.list_models(
            order=Order.asc,
            limit=10,
        )

        # Verify the call
        mock_reranker.list_models.assert_called_once_with(
            order=Order.asc,
            limit=10,
        )

        # Check response
        assert len(response.models) == 2
        assert response.models[0].identifier == "rerank-model-1"
        assert response.models[1].metadata["max_documents"] == 1000

    def test_rerank_result_validation(self):
        """Test RerankResult model validation."""
        # Valid result
        result = RerankResult(
            index=0,
            relevance_score=0.95,
            document="Test document",
        )
        assert result.index == 0
        assert result.relevance_score == 0.95
        assert result.document == "Test document"

        # Result without document
        result_no_doc = RerankResult(
            index=1,
            relevance_score=0.85,
        )
        assert result_no_doc.document is None

    def test_rerank_response_validation(self):
        """Test RerankResponse model validation."""
        results = [
            RerankResult(index=0, relevance_score=0.9),
            RerankResult(index=1, relevance_score=0.8),
        ]

        # Basic response
        response = RerankResponse(
            results=results,
            model="test-model",
        )
        assert len(response.results) == 2
        assert response.model == "test-model"
        assert response.usage is None

        # Response with usage
        response_with_usage = RerankResponse(
            results=results,
            model="test-model",
            usage={"tokens": 100},
        )
        assert response_with_usage.usage == {"tokens": 100}
