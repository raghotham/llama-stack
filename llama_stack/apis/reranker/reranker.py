# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.common.responses import Order
from llama_stack.apis.models import Model
from llama_stack.apis.telemetry import MetricResponseMixin
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class RerankResult(BaseModel):
    """Result of reranking a single document.

    :param index: Original index of the document in the input list
    :param relevance_score: Relevance score (typically 0.0 to 1.0)
    :param document: Original document text if return_documents=True
    """

    index: int = Field(..., description="Original index of the document")
    relevance_score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    document: str | None = Field(None, description="Original document text if requested")


@json_schema_type
class RerankRequest(BaseModel):
    """Request for reranking documents.

    :param query: The search query
    :param documents: List of documents to rerank
    :param model: Model identifier for reranking
    :param top_n: Return only top N results (default: return all)
    :param truncation: Auto-truncate documents to fit model context
    :param return_documents: Include original documents in response
    """

    query: str = Field(..., description="The search query")
    documents: list[str] = Field(..., description="List of documents to rerank")
    model: str = Field(..., description="Model identifier for reranking")
    top_n: int | None = Field(None, description="Return only top N results", ge=1)
    truncation: bool = Field(True, description="Auto-truncate documents to fit model context")
    return_documents: bool = Field(False, description="Include original documents in response")


@json_schema_type
class RerankResponse(MetricResponseMixin):
    """Response from reranking operation.

    :param results: List of reranked results ordered by relevance
    :param model: Model used for reranking
    :param usage: Token usage information (optional)
    """

    results: list[RerankResult] = Field(..., description="Ordered results by relevance")
    model: str = Field(..., description="Model used for reranking")
    usage: dict | None = Field(None, description="Token usage information")


@json_schema_type
class ListModelsRequest(BaseModel):
    """Request to list available reranker models.

    :param order: Sort order for the returned models
    :param limit: Maximum number of models to return
    """

    order: Order = Field(default=Order.asc)
    limit: int = Field(default=100)


@json_schema_type
class ListModelsResponse(BaseModel):
    """Response containing available reranker models.

    :param models: List of available reranker models
    """

    models: list[Model]


@runtime_checkable
@trace_protocol
class Reranker(Protocol):
    """Protocol for reranking documents based on relevance to a query."""

    @webmethod(route="/reranker/rerank")
    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int | None = None,
        truncation: bool = True,
        return_documents: bool = False,
    ) -> RerankResponse:
        """Rerank documents based on relevance to a query.

        Args:
            query: The search query
            documents: List of documents to rerank
            model: Model identifier for reranking
            top_n: Return only top N results (default: return all)
            truncation: Auto-truncate documents to fit model context
            return_documents: Include original documents in response

        Returns:
            RerankResponse with ordered results by relevance
        """
        ...

    @webmethod(route="/reranker/list-models")
    async def list_models(
        self,
        order: Order = Order.asc,
        limit: int = 100,
    ) -> ListModelsResponse:
        """List available reranker models.

        Args:
            order: Sort order for the returned models
            limit: Maximum number of models to return

        Returns:
            ListModelsResponse with available models
        """
        ...
