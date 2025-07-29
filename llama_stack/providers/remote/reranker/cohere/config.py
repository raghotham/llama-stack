# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class CohereProviderDataValidator(BaseModel):
    cohere_api_key: str | None = Field(
        default=None,
        description="API key for Cohere models",
    )


@json_schema_type
class CohereConfig(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="API key for Cohere models",
    )
    api_base_url: str | None = Field(
        default="https://api.cohere.com/v2",
        description="Base URL for Cohere API",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.COHERE_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "api_key": api_key,
        }
