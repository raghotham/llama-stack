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

from .config import CohereConfig

COHERE_SUPPORTED_MODELS: dict[str, dict[str, int | str]] = {
    "rerank-v3.5": {
        "display_name": "Cohere Rerank v3.5",
        "max_tokens_per_doc": 4096,
    },
    "rerank-v3": {
        "display_name": "Cohere Rerank v3",
        "max_tokens_per_doc": 4096,
    },
    "rerank-multilingual-v3": {
        "display_name": "Cohere Rerank Multilingual v3",
        "max_tokens_per_doc": 4096,
    },
    "rerank-english-v3": {
        "display_name": "Cohere Rerank English v3",
        "max_tokens_per_doc": 4096,
    },
}


@trace_runtime
class CohereReranker(Reranker):
    def __init__(self, config: CohereConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key is required")

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
        if model not in COHERE_SUPPORTED_MODELS:
            raise ValueError(f"Model {model} is not supported by Cohere")

        # Cohere API limits
        if len(documents) > 1000:
            raise ValueError("Cohere rerank API supports up to 1000 documents")

        request_data: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
        }

        if top_n is not None:
            request_data["top_n"] = top_n

        # Handle truncation based on model limits
        if truncation:
            model_info = COHERE_SUPPORTED_MODELS[model]
            max_tokens_value = model_info["max_tokens_per_doc"]
            if not isinstance(max_tokens_value, int):
                raise ValueError(f"Invalid max_tokens_per_doc value for model {model}")
            request_data["max_tokens_per_doc"] = max_tokens_value

        try:
            response = await self.client.post(
                f"{self.api_base_url}/rerank",
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Cohere API error: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Cohere API: {str(e)}") from e

        # Parse response
        results = []
        for result in data["results"]:
            rerank_result = RerankResult(
                index=result["index"],
                relevance_score=result["relevance_score"],
                document=documents[result["index"]] if return_documents else None,
            )
            results.append(rerank_result)

        usage = None
        if "meta" in data and "billed_units" in data["meta"]:
            usage = {"search_units": data["meta"]["billed_units"]["search_units"]}

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
        for model_id, model_info in COHERE_SUPPORTED_MODELS.items():
            models.append(
                Model(
                    identifier=model_id,
                    provider_id="cohere",
                    metadata={
                        "display_name": model_info["display_name"],
                        "max_tokens_per_doc": model_info["max_tokens_per_doc"],
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
