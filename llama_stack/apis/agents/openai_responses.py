# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type, register_schema

# NOTE(ashwin): this file is literally a copy of the OpenAI responses API schema. We should probably
# take their YAML and generate this file automatically. Their YAML is available.


@json_schema_type
class OpenAIResponseError(BaseModel):
    code: str
    message: str


@json_schema_type
class OpenAIResponseInputMessageContentText(BaseModel):
    text: str
    type: Literal["input_text"] = "input_text"


@json_schema_type
class OpenAIResponseInputMessageContentImage(BaseModel):
    detail: Literal["low"] | Literal["high"] | Literal["auto"] = "auto"
    type: Literal["input_image"] = "input_image"
    # TODO: handle file_id
    image_url: str | None = None


# TODO: handle file content types
OpenAIResponseInputMessageContent = Annotated[
    OpenAIResponseInputMessageContentText | OpenAIResponseInputMessageContentImage,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputMessageContent, name="OpenAIResponseInputMessageContent")


@json_schema_type
class OpenAIResponseOutputMessageContentOutputText(BaseModel):
    text: str
    type: Literal["output_text"] = "output_text"


OpenAIResponseOutputMessageContent = Annotated[
    OpenAIResponseOutputMessageContentOutputText,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutputMessageContent, name="OpenAIResponseOutputMessageContent")


@json_schema_type
class OpenAIResponseMessage(BaseModel):
    """
    Corresponds to the various Message types in the Responses API.
    They are all under one type because the Responses API gives them all
    the same "type" value, and there is no way to tell them apart in certain
    scenarios.
    """

    content: str | list[OpenAIResponseInputMessageContent] | list[OpenAIResponseOutputMessageContent]
    role: Literal["system"] | Literal["developer"] | Literal["user"] | Literal["assistant"]
    type: Literal["message"] = "message"

    # The fields below are not used in all scenarios, but are required in others.
    id: str | None = None
    status: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageWebSearchToolCall(BaseModel):
    id: str
    status: str
    type: Literal["web_search_call"] = "web_search_call"


@json_schema_type
class OpenAIResponseOutputMessageFunctionToolCall(BaseModel):
    call_id: str
    name: str
    arguments: str
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    status: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPCall(BaseModel):
    id: str
    type: Literal["mcp_call"] = "mcp_call"
    arguments: str
    name: str
    server_label: str
    error: str | None = None
    output: str | None = None


class MCPListToolsTool(BaseModel):
    input_schema: dict[str, Any]
    name: str
    description: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPListTools(BaseModel):
    id: str
    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    server_label: str
    tools: list[MCPListToolsTool]


OpenAIResponseOutput = Annotated[
    OpenAIResponseMessage
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutput, name="OpenAIResponseOutput")


@json_schema_type
class OpenAIResponseObject(BaseModel):
    created_at: int
    error: OpenAIResponseError | None = None
    id: str
    model: str
    object: Literal["response"] = "response"
    output: list[OpenAIResponseOutput]
    parallel_tool_calls: bool = False
    previous_response_id: str | None = None
    status: str
    temperature: float | None = None
    top_p: float | None = None
    truncation: str | None = None
    user: str | None = None


@json_schema_type
class OpenAIResponseObjectStreamResponseCreated(BaseModel):
    response: OpenAIResponseObject
    type: Literal["response.created"] = "response.created"


@json_schema_type
class OpenAIResponseObjectStreamResponseCompleted(BaseModel):
    response: OpenAIResponseObject
    type: Literal["response.completed"] = "response.completed"


OpenAIResponseObjectStream = Annotated[
    OpenAIResponseObjectStreamResponseCreated | OpenAIResponseObjectStreamResponseCompleted,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseObjectStream, name="OpenAIResponseObjectStream")


@json_schema_type
class OpenAIResponseInputFunctionToolCallOutput(BaseModel):
    """
    This represents the output of a function call that gets passed back to the model.
    """

    call_id: str
    output: str
    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    status: str | None = None


OpenAIResponseInput = Annotated[
    # Responses API allows output messages to be passed in as input
    OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseInputFunctionToolCallOutput
    |
    # Fallback to the generic message type as a last resort
    OpenAIResponseMessage,
    Field(union_mode="left_to_right"),
]
register_schema(OpenAIResponseInput, name="OpenAIResponseInput")


@json_schema_type
class OpenAIResponseInputToolWebSearch(BaseModel):
    type: Literal["web_search"] | Literal["web_search_preview_2025_03_11"] = "web_search"
    # TODO: actually use search_context_size somewhere...
    search_context_size: str | None = Field(default="medium", pattern="^low|medium|high$")
    # TODO: add user_location


@json_schema_type
class OpenAIResponseInputToolFunction(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None
    strict: bool | None = None


class FileSearchRankingOptions(BaseModel):
    ranker: str | None = None
    score_threshold: float | None = Field(default=0.0, ge=0.0, le=1.0)


@json_schema_type
class OpenAIResponseInputToolFileSearch(BaseModel):
    type: Literal["file_search"] = "file_search"
    vector_store_id: list[str]
    ranking_options: FileSearchRankingOptions | None = None
    # TODO: add filters


class ApprovalFilter(BaseModel):
    always: list[str] | None = None
    never: list[str] | None = None


class AllowedToolsFilter(BaseModel):
    tool_names: list[str] | None = None


@json_schema_type
class OpenAIResponseInputToolMCP(BaseModel):
    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str
    headers: dict[str, Any] | None = None

    require_approval: Literal["always"] | Literal["never"] | ApprovalFilter = "never"
    allowed_tools: list[str] | AllowedToolsFilter | None = None


OpenAIResponseInputTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseInputToolMCP,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputTool, name="OpenAIResponseInputTool")


class ListOpenAIResponseInputItem(BaseModel):
    data: list[OpenAIResponseInput]
    object: Literal["list"] = "list"


@json_schema_type
class OpenAIResponseObjectWithInput(OpenAIResponseObject):
    input: list[OpenAIResponseInput]


@json_schema_type
class ListOpenAIResponseObject(BaseModel):
    data: list[OpenAIResponseObjectWithInput]
    has_more: bool
    first_id: str
    last_id: str
    object: Literal["list"] = "list"
