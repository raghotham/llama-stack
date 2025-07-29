# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        remote_provider_spec(
            api=Api.reranker,
            adapter=AdapterSpec(
                adapter_type="cohere",
                pip_packages=[
                    "httpx",
                ],
                module="llama_stack.providers.remote.reranker.cohere",
                config_class="llama_stack.providers.remote.reranker.cohere.CohereConfig",
                description="Cohere's reranking models for improving search relevance.",
            ),
        ),
        remote_provider_spec(
            api=Api.reranker,
            adapter=AdapterSpec(
                adapter_type="voyage",
                pip_packages=[
                    "httpx",
                ],
                module="llama_stack.providers.remote.reranker.voyage",
                config_class="llama_stack.providers.remote.reranker.voyage.VoyageConfig",
                description="Voyage AI's multilingual reranking models for cross-lingual search.",
            ),
        ),
        remote_provider_spec(
            api=Api.reranker,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=[
                    "httpx",
                ],
                module="llama_stack.providers.remote.reranker.nvidia",
                config_class="llama_stack.providers.remote.reranker.nvidia.NvidiaConfig",
                description="NVIDIA's reranking models optimized for GPU inference.",
            ),
        ),
    ]
