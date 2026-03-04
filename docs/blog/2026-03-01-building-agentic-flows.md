---
slug: building-agentic-flows-with-conversations-and-responses
title: "Building a Self-Improving Agent with Llama Stack"
authors:
  - name: Raghotham Murthy
    title: Llama Stack Team
    url: https://github.com/raghotham
    image_url: https://github.com/raghotham.png
tags: [agents, responses-api, conversations, prompts, tutorial]
date: 2026-03-01
---

What if your AI agent could improve itself? Most agent tutorials show a single loop — user asks a question, the agent calls some tools, returns an answer. But what happens when you need to systematically improve your agent's behavior over time?

In this post, we build an RL-style optimization loop where an outer "optimizer agent" iteratively improves the system prompt of an inner RAG agent — using Llama Stack's own APIs as tools. The optimizer proposes prompt changes, tests them against a benchmark, scores the results with an LLM judge, and learns from its history across iterations.

This is self-referential: **a Llama Stack agent improving another Llama Stack agent**, using the Responses API, Conversations API, Prompts API, and Vector Stores as its toolkit.

<!--truncate-->

## What We're Building

The system has two agents, each built on Llama Stack but serving very different roles:

- **RAGAgent** (inner): A question-answering agent that uses `file_search` to retrieve documents from a vector store and generate answers. Its system prompt controls how it uses context, cites sources, and structures responses — and that system prompt is the thing being optimized.
- **OptimizerAgent** (outer): An RL-style agent that iteratively proposes better system prompts, runs the RAG agent against a test suite, scores the results with an LLM judge, and records what worked and what didn't.

The optimizer uses the Responses API with client-side function tools that call back into Llama Stack's Prompts API and Responses API. The Conversations API ties it all together — by passing a `conversation` ID to each `responses.create()` call, the optimizer's full reasoning history persists across iterations. It remembers which prompts it already tried, what scores they received, and what its reasoning was, so it can make progressively better decisions.

```
┌─────────────────────────────────────────────────────┐
│  OptimizerAgent (Responses API + function tools)    │
│                                                     │
│  for each iteration:                                │
│    1. get_history()        → read past scores       │
│    2. get_prompt()         → read current prompt    │
│    3. update_prompt()      → write improved prompt  │
│    4. run_rag_test()       → test the RAG agent     │
│    5. judge_answer()       → score with LLM judge   │
│    6. log_result()         → record the score       │
│                                                     │
│  Conversation tracks reasoning across iterations    │
├─────────────────────────────────────────────────────┤
│  RAGAgent (Responses API + file_search)             │
│  Vector Store ◄── file_search                       │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

- A running Llama Stack server with Ollama: `llama stack run ollama`
- Python SDK: `pip install llama-stack-client`
- Two models via Ollama: `llama3.1:8b` for the RAG agent and `gpt-oss:20b` as the judge

## The Inner Agent: RAGAgent

The RAG agent is simple — it takes a question and a system prompt, searches a vector store using `file_search`, and returns an answer. The Responses API handles the retrieval and generation in a single call:

```python
class RAGAgent:
    def __init__(self, client: LlamaStackClient, model: str, vector_store_id: str):
        self.client = client
        self.model = model
        self.vector_store_id = vector_store_id

    @classmethod
    def from_files(cls, client, model, name, file_paths):
        """Create a RAGAgent with a new vector store populated from local files.
        Also accepts embedding_model and embedding_dimension kwargs."""
        vector_store = client.vector_stores.create(name=name)
        for path in file_paths:
            file = client.files.create(file=open(path, "rb"), purpose="assistants")
            client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file.id
            )
            # ... poll until indexing completes ...
        return cls(client, model, vector_store.id)

    def query(self, question: str, system_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=question,
            instructions=system_prompt,
            tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
            stream=False,
        )
        return response.output_text
```

The `from_files` classmethod handles vector store creation, file upload, and indexing. The `instructions` parameter sets the system prompt, and `file_search` tells the Responses API to search the vector store for relevant context before generating. The system prompt is what the optimizer will iterate on — how the agent uses retrieved context, whether it cites sources, how concise it is.

## The Outer Agent: OptimizerAgent

The optimizer agent is itself an LLM-driven loop — it uses the Responses API with function tools, just like any other Llama Stack agent. What makes it interesting is *what* those tools do: they call back into Llama Stack's own APIs.

The optimizer doesn't have hardcoded logic for "try this, then that." Instead, it receives a high-level instruction ("improve this RAG agent's system prompt") and a set of tools, and the LLM decides which tools to call and in what order. Over multiple iterations, the Conversations API preserves the optimizer's full reasoning history — it can see which prompts it already tried, what scores they got, and why. This is the RL-style feedback loop: propose → test → score → learn → repeat.

The tools fall into three categories:

### Prompt management tools (Prompts API)

The optimizer reads and writes system prompt versions through the Prompts API. Each version is immutable — calling `update` creates a new version rather than overwriting the old one. This gives us a full audit trail of every prompt the optimizer tried:

```python
class OptimizerAgent:
    ...

    def _register_tools(self):

        @tool
        def update_prompt(
            new_prompt: Annotated[str, "The improved system prompt text"],
            current_version: Annotated[
                int, "Current version number (for optimistic locking)"
            ],
        ) -> dict:
            """Create a new version of the system prompt."""
            result = self.client.prompts.update(
                self.prompt_id,
                prompt=new_prompt,
                version=current_version,
            )
            return {"prompt_id": result.prompt_id, "version": result.version}

        @tool
        def get_prompt(
            version: Annotated[int | None, "Specific version to fetch"] = None,
        ) -> dict:
            """Fetch the current system prompt text and version."""
            result = self.client.prompts.retrieve(self.prompt_id, version=version)
            return {"prompt": result.prompt, "version": result.version}
```

The `current_version` parameter provides optimistic locking — if another process updated the prompt between our read and write, the update will fail rather than silently overwrite.

### Testing and judging tools

These tools close the feedback loop. `run_rag_test` calls into the inner RAG agent (which itself uses the Responses API with `file_search`), and `judge_answer` uses the Responses API with a separate, stronger model to score the output. This separation matters — the judge model should be more capable than the model being evaluated:

```python
class OptimizerAgent:
    ...

    def _register_tools(self):
        ...

        @tool
        def run_rag_test(
            question: Annotated[str, "The test question to ask the RAG agent"],
            system_prompt: Annotated[str, "The system prompt to use"],
        ) -> str:
            """Run the inner RAG agent on a test question."""
            return self.rag_agent.query(question, system_prompt)

        @tool
        def judge_answer(
            question: Annotated[str, "The original question"],
            expected: Annotated[str, "The expected answer"],
            actual: Annotated[str, "The RAG agent's actual answer"],
        ) -> dict:
            """Score a RAG answer using LLM-as-judge."""
            response = self.client.responses.create(
                model=self.judge_model,
                input=(
                    f"Score the following answer on a scale of 0.0 to 1.0.\n\n"
                    f"Question: {question}\nExpected: {expected}\nActual: {actual}\n\n"
                    f'Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}'
                ),
                stream=False,
            )
            return json.loads(response.output_text)
```

Notice the self-referential structure: the optimizer agent calls `run_rag_test`, which calls `responses.create()` on the inner agent, which triggers `file_search` on the vector store — three layers of Llama Stack APIs invoked from a single tool call.

### Score ledger tools

The optimizer also has `log_result` and `get_history` tools backed by a `ScoreLedger` — a simple SQLite table that maps `(prompt_id, version)` to scores and reasoning. The Prompts API stores the prompt text and versions; the ledger tracks how well each version performed. This separation keeps concerns clean — prompt storage is Llama Stack's job, evaluation tracking is ours. See the [full implementation](./prompt_optimizer_sketch.py) for details.

## The Optimization Loop

This is where everything comes together. Each iteration, we give the optimizer a high-level instruction ("improve this RAG agent's system prompt") and let it decide which tools to call. The inner `while True` loop is the standard Responses API agentic pattern — keep calling `responses.create()` until the model stops emitting tool calls. The outer `for` loop runs multiple iterations, and the `conversation` parameter ensures the optimizer's reasoning accumulates across all of them:

```python
class OptimizerAgent:
    ...

    def run(self, max_iterations: int = 5):
        for iteration in range(max_iterations):
            inputs = [{"role": "user", "content": optimization_prompt}]

            # Agentic loop: keep going until the model stops calling tools
            while True:
                response = self.client.responses.create(
                    model=self.model,
                    input=inputs,
                    tools=tool_schemas,
                    conversation=self.conversation_id,
                    stream=False,
                )

                function_calls = [
                    o for o in response.output if o.type == "function_call"
                ]
                if not function_calls:
                    break  # Model is done — no more tool calls

                # Execute tool calls and feed results back
                inputs = []
                for fc in function_calls:
                    result = self._execute_tool_call(fc.name, fc.arguments)
                    inputs.append(fc)
                    inputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc.call_id,
                            "output": result,
                        }
                    )
```

The agentic pattern here works the same way for any Responses API agent:

1. Call `responses.create()` with function tools and a `conversation` ID
2. Check if the response contains `function_call` outputs — if not, the model is done
3. Execute each tool call client-side and collect the results
4. Feed the results back as `function_call_output` items in the next request
5. Repeat until the model responds with text instead of tool calls

The key insight is the `conversation` parameter. Without it, each iteration would start from scratch — the optimizer would have no memory of what it already tried. With it, Llama Stack persists all turns server-side, so by iteration 3, the optimizer can look back at its history and reason: "v1 scored 0.4 because answers were too vague, v2 scored 0.7 after I added citation instructions, let me try adding format constraints for v3."

## Running It

First, make sure you have a Llama Stack server running with Ollama:

```bash
ollama pull llama3.1:8b
ollama pull gpt-oss:20b
uv run --with llama-stack llama stack run ollama
```

Then set up the vector store with some documents for the RAG agent to search over, create the initial prompt, and run the optimizer. You can use any text file — here we use the [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) as an example:

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")
MODEL = "ollama/llama3.1:8b"

# Create the inner RAG agent with a vector store from local files
rag_agent = RAGAgent.from_files(
    client,
    model=MODEL,
    name="llama-docs",
    file_paths=["llama3_model_card.txt"],
)

# Verify the RAG agent works
answer = rag_agent.query(
    "What is the max context length of Llama 3.1?",
    system_prompt="Answer based on the provided context.",
)
print(f"RAG agent says: {answer}")

# Create the initial system prompt via Prompts API
initial = client.prompts.create(
    prompt="You are a helpful assistant. Answer questions based on the provided context.",
)

# Run the optimizer
optimizer = OptimizerAgent(
    client=client,
    model=MODEL,
    judge_model="ollama/gpt-oss:20b",
    rag_agent=rag_agent,
    ledger=ScoreLedger(),
    prompt_id=initial.prompt_id,
    test_cases=[
        {
            "question": "What is the max context length of Llama 3.1?",
            "expected": "128K tokens",
        },
        {
            "question": "What languages does Llama 3.1 support?",
            "expected": "English, German, French, Italian, Portuguese, Hindi, Spanish, Thai",
        },
    ],
)
optimizer.run(max_iterations=5)

# Show the best prompt
best = optimizer.best_prompt()
print(f"Best prompt (v{best['version']}, score={best['score']:.2f}):")
print(f"  {best['prompt']}")
```

The full implementation with tool schema generation and all supporting code is available at [prompt_optimizer_sketch.py](./prompt_optimizer_sketch.py).

## How It Works Under the Hood

The Responses API supports two kinds of tools, and this example uses both:

- **Server-side tools** like `file_search` and `web_search` are executed automatically by the Responses API — when the RAG agent calls `file_search`, the server searches the vector store, retrieves relevant chunks, and feeds them back to the model without any client-side code. This is what makes the inner RAG agent so simple.
- **Client-side function tools** return tool call objects for you to execute yourself, which is what the optimizer's `while True` loop handles. You execute the function, then send the result back as a `function_call_output`.

The `conversation` parameter persists all turns server-side via the Conversations API, so the optimizer sees its full reasoning history without manual state management. And the Prompts API auto-increments versions on each update, with optimistic locking via `current_version` to prevent silent overwrites — important when you're running multiple optimization experiments.

## What's Next

The pattern here — an LLM-driven loop that uses Llama Stack APIs as tools — generalizes well beyond prompt optimization. The same architecture works for any scenario where an agent needs to manage, test, and iterate on other agents or resources:

- **MCP tools** for connecting to external services (databases, APIs, code execution sandboxes) — the optimizer could test how well the RAG agent integrates external data
- **Web search** alongside `file_search` for agents that combine local knowledge with live web results
- **Multiple RAG agents** with different vector stores, optimized in parallel — the optimizer could manage a fleet of specialized agents

To learn more:
- [Responses API documentation](/docs/building_applications/responses_vs_agents)
- [Conversations API documentation](/docs/api-openai/conformance#conversations)
- [OpenAI API compatibility](/docs/api-openai)
- [Vector Stores documentation](/docs/building_applications/rag)
- [Join our Discord](https://discord.gg/llama-stack)
