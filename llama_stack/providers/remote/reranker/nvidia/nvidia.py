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

from .config import NvidiaConfig

NVIDIA_SUPPORTED_MODELS: dict[str, dict[str, int | str]] = {
    "nvidia/nv-rerankqa-mistral-4b-v3": {
        "display_name": "NVIDIA Rerank QA Mistral 4B v3",
        "max_documents": 100,
        "max_input_length": 512,
    },
    "nvidia/llama-3_2-nv-rerankqa-1b-v1": {
        "display_name": "NVIDIA Llama 3.2 Rerank QA 1B v1",
        "max_documents": 100,
        "max_input_length": 512,
    },
}


@trace_runtime
class NvidiaReranker(Reranker):
    def __init__(self, config: NvidiaConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key is required")

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
        if model not in NVIDIA_SUPPORTED_MODELS:
            raise ValueError(f"Model {model} is not supported by NVIDIA")

        model_info = NVIDIA_SUPPORTED_MODELS[model]

        # Validate document count
        max_docs_value = model_info["max_documents"]
        if not isinstance(max_docs_value, int):
            raise ValueError(f"Invalid max_documents value for model {model}")
        if len(documents) > max_docs_value:
            raise ValueError(f"NVIDIA {model} supports up to {max_docs_value} documents")

        # NVIDIA expects passages instead of documents
        passages = [{"text": doc} for doc in documents]

        request_data: dict[str, Any] = {
            "model": model,
            "query": {"text": query},
            "passages": passages,
        }

        if top_n is not None:
            request_data["top_n"] = top_n

        if truncation:
            request_data["truncate"] = "END"  # NVIDIA specific truncation

        try:
            response = await self.client.post(
                f"{self.api_base_url}/ranking",
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"NVIDIA API error: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to connect to NVIDIA API: {str(e)}") from e

        # Parse response
        results = []
        rankings = data.get("rankings", [])

        for ranking in rankings:
            rerank_result = RerankResult(
                index=ranking["index"],
                relevance_score=ranking["logit"],  # NVIDIA uses logit scores
                document=documents[ranking["index"]] if return_documents else None,
            )
            results.append(rerank_result)

        usage = None
        if "usage" in data:
            usage = {
                "total_tokens": data["usage"].get("total_tokens"),
                "prompt_tokens": data["usage"].get("prompt_tokens"),
            }

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
        for model_id, model_info in NVIDIA_SUPPORTED_MODELS.items():
            models.append(
                Model(
                    identifier=model_id,
                    provider_id="nvidia",
                    metadata={
                        "display_name": model_info["display_name"],
                        "max_documents": model_info["max_documents"],
                        "max_input_length": model_info["max_input_length"],
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
