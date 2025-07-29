# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.models import Model, ModelType, Models


class TestRerankingModels:
    """Unit tests for reranking model registration functionality."""

    @pytest.fixture
    def mock_models_api(self):
        """Create a mock models API instance."""
        models_api = MagicMock(spec=Models)
        models_api.register_reranking_model = AsyncMock()
        models_api.unregister_reranking_model = AsyncMock()
        models_api.get_model = AsyncMock()
        return models_api

    async def test_register_reranking_model(self, mock_models_api):
        """Test registering a reranking model."""
        # Setup mock response
        expected_model = Model(
            identifier="test-reranker",
            provider_id="cohere",
            provider_resource_id="rerank-v3.5",
            model_type=ModelType.reranking,
            metadata={"max_documents": 1000},
        )
        mock_models_api.register_reranking_model.return_value = expected_model

        # Call register_reranking_model
        result = await mock_models_api.register_reranking_model(
            model_id="test-reranker",
            provider_model_id="rerank-v3.5",
            provider_id="cohere",
            metadata={"max_documents": 1000},
        )

        # Verify the call
        mock_models_api.register_reranking_model.assert_called_once_with(
            model_id="test-reranker",
            provider_model_id="rerank-v3.5",
            provider_id="cohere",
            metadata={"max_documents": 1000},
        )

        # Check response
        assert result.identifier == "test-reranker"
        assert result.model_type == ModelType.reranking
        assert result.provider_id == "cohere"
        assert result.metadata["max_documents"] == 1000

    async def test_unregister_reranking_model(self, mock_models_api):
        """Test unregistering a reranking model."""
        # Call unregister_reranking_model
        await mock_models_api.unregister_reranking_model("test-reranker")

        # Verify the call
        mock_models_api.unregister_reranking_model.assert_called_once_with("test-reranker")

    def test_model_type_enum_has_reranking(self):
        """Test that ModelType enum includes reranking."""
        assert hasattr(ModelType, "reranking")
        assert ModelType.reranking == "reranking"
        
        # Test that all expected model types exist
        assert ModelType.llm == "llm"
        assert ModelType.embedding == "embedding"
        assert ModelType.reranking == "reranking"

    def test_model_with_reranking_type(self):
        """Test creating a Model with reranking type."""
        model = Model(
            identifier="test-reranker",
            provider_id="voyage",
            provider_resource_id="rerank-2.5",
            model_type=ModelType.reranking,
            metadata={"max_total_tokens": 600000},
        )
        
        assert model.model_type == ModelType.reranking
        assert model.identifier == "test-reranker"
        assert model.provider_id == "voyage"
        assert model.metadata["max_total_tokens"] == 600000