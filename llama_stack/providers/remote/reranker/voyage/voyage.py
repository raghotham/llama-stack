# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

import httpx

from llama_stack.apis.common.responses import Order
from llama_stack.apis.models import Model
from llama_stack.apis.reranker import (
    ListModelsResponse,
    Reranker,
    RerankResponse,
    RerankResult,
)
from llama_stack.utils.telemetry import trace_runtime

from .config import VoyageConfig

VOYAGE_SUPPORTED_MODELS: dict[str, dict[str, int | str]] = {
    "rerank-2.5": {
        "display_name": "Voyage Rerank 2.5",
        "max_query_tokens": 8000,
        "max_total_tokens": 600000,
        "max_documents": 1000,
    },
    "rerank-2.5-lite": {
        "display_name": "Voyage Rerank 2.5 Lite",
        "max_query_tokens": 8000,
        "max_total_tokens": 400000,
        "max_documents": 1000,
    },
    "rerank-multilingual-2": {
        "display_name": "Voyage Rerank Multilingual 2",
        "max_query_tokens": 2000,
        "max_total_tokens": 300000,
        "max_documents": 1000,
    },
}


@trace_runtime
class VoyageReranker(Reranker):
    def __init__(self, config: VoyageConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key is required")

        self.api_base_url = config.api_base_url
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int | None = None,
        truncation: bool = True,
        return_documents: bool = False,
    ) -> RerankResponse:
        if model not in VOYAGE_SUPPORTED_MODELS:
            raise ValueError(f"Model {model} is not supported by Voyage AI")

        model_info = VOYAGE_SUPPORTED_MODELS[model]

        # Validate document count
        max_docs_value = model_info["max_documents"]
        if not isinstance(max_docs_value, int):
            raise ValueError(f"Invalid max_documents value for model {model}")
        if len(documents) > max_docs_value:
            raise ValueError(f"Voyage {model} supports up to {max_docs_value} documents")

        request_data: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "truncation": truncation,
        }

        if top_n is not None:
            request_data["top_k"] = top_n  # Voyage uses top_k

        try:
            response = await self.client.post(
                f"{self.api_base_url}/rerank",
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Voyage AI API error: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Voyage AI API: {str(e)}") from e

        # Parse response
        results = []
        for result in data["results"]:
            rerank_result = RerankResult(
                index=result["index"],
                relevance_score=result["relevance_score"],
                document=result.get("document") if return_documents else None,
            )
            results.append(rerank_result)

        usage = None
        if "total_tokens" in data:
            usage = {"total_tokens": data["total_tokens"]}

        return RerankResponse(
            results=results,
            model=model,
            usage=usage,
        )

    async def list_models(
        self,
        order: Order = Order.asc,
        limit: int = 100,
    ) -> ListModelsResponse:
        models = []
        for model_id, model_info in VOYAGE_SUPPORTED_MODELS.items():
            models.append(
                Model(
                    identifier=model_id,
                    provider_id="voyage",
                    metadata={
                        "display_name": model_info["display_name"],
                        "max_query_tokens": model_info["max_query_tokens"],
                        "max_total_tokens": model_info["max_total_tokens"],
                        "max_documents": model_info["max_documents"],
                    },
                )
            )

        # Apply ordering
        if order == Order.desc:
            models.reverse()

        # Apply limit
        models = models[:limit]

        return ListModelsResponse(models=models)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
