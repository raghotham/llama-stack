---
slug: building-agentic-flows-with-conversations-and-responses
title: "Building a Self-Improving Agent with Llama Stack"
authors:
  - name: Raghotham Murthy
    title: Llama Stack Core Team
    url: https://github.com/raghotham
    image_url: https://github.com/raghotham.png
tags: [agents, responses-api, conversations, prompts, tutorial]
date: 2026-03-01
---

What if your AI agent could improve itself? Most agent tutorials show a single loop — user asks a question, the agent calls some tools, returns an answer. But what happens when you need to systematically improve your agent's behavior over time?

In this post, we build a two-tier system: an inner **ResearchAgent** that uses the Responses API agentic loop to answer questions from an internal engineering knowledge base, and an outer **optimization loop** that iteratively improves the agent's system prompt. The research agent is the genuinely agentic component — it decides what to search, which local files to read, and which documents to index into the vector store. The optimizer is just a deterministic Python `for` loop that orchestrates evaluation and prompt updates.

This is self-referential: **a Llama Stack agent improving itself**, using the Responses API, Prompts API, and Vector Stores as its toolkit.

<!--truncate-->

## What We're Building

The system has two tiers with very different roles:

- **ResearchAgent** (inner, agentic): A research agent that uses the Responses API `while True` loop with server-side `file_search` and client-side function tools (`read_local_file`, `index_document`, `list_local_files`). The agent decides what to search in the vector store, discovers unindexed local files, reads them, indexes the relevant ones, and searches again with the enriched knowledge base. Its system prompt is the thing being optimized.
- **`optimize_prompt()`** (outer, deterministic): A plain Python function with a `for` loop. Each iteration reads the current prompt, runs the research agent on test cases, judges the answers, logs scores, proposes an improved prompt, and saves it. No LLM-driven tool selection — just sequential function calls.

The agentic pattern lives where it genuinely belongs: in the research agent, where the LLM needs to make real decisions about what to search, read, and index. The optimization loop is always the same sequence of steps, so it's just code.

```
┌──────────────────────────────────────────────────────┐
│  optimize_prompt() (deterministic Python loop)       │
│                                                      │
│  for each iteration:                                 │
│    1. Read current prompt (Prompts API)              │
│    2. Run research agent on all test cases           │
│    3. Judge answers (Responses API)                  │
│    4. Log scores (SQLite ledger)                     │
│    5. Propose new prompt (Responses API)             │
│    6. Save new version (Prompts API)                 │
├──────────────────────────────────────────────────────┤
│  ResearchAgent (Responses API agentic loop)          │
│                                                      │
│  Server-side:  file_search → Vector Store            │
│  Client-side:  read_local_file, index_document,      │
│                list_local_files                       │
│  Agent decides what to search, read, and index       │
└──────────────────────────────────────────────────────┘
```

## Prerequisites

- A running Llama Stack server with Ollama: `uv run --with llama-stack llama stack run ollama`
- Python SDK: `uv pip install llama-stack-client`
- Two models via Ollama: `llama3.1:8b` for the research agent and `gpt-oss:20b` as the judge

## The Inner Agent: ResearchAgent

The research agent is the heart of the system — and the showcase for the Responses API agentic pattern. Unlike a simple single-call RAG agent, it has real decisions to make: the vector store might not have enough context, so the agent can discover local files, read them, index the relevant ones, and search again.

It has one server-side tool and three client-side function tools:

- **`file_search`** (server-side): Searches the vector store for relevant documents. The Responses API executes this automatically — no client code needed.
- **`read_local_file(path)`**: Reads an unindexed local file (e.g., a newly written postmortem not yet in the knowledge base).
- **`index_document(file_path)`**: Uploads a file to the vector store via the Files API and `vector_stores.files.create()`. This is the key insight: the agent actively curates the knowledge base.
- **`list_local_files(directory)`**: Discovers available `.md` and `.txt` files in a directory.

The `query()` method is the standard Responses API agentic loop — keep calling `responses.create()` until the model stops emitting tool calls:

```python
class ResearchAgent:
    def __init__(self, client, model, vector_store_id, local_docs_dir=None):
        self.client = client
        self.model = model
        self.vector_store_id = vector_store_id
        self.local_docs_dir = local_docs_dir
        self._tools = {
            "read_local_file": self._read_local_file,
            "index_document": self._index_document,
            "list_local_files": self._list_local_files,
        }

    def query(self, question: str, system_prompt: str) -> str:
        """Agentic loop: search, read local files, index, repeat."""
        inputs = question
        tools = self._tool_schemas()

        while True:
            response = self.client.responses.create(
                model=self.model,
                input=inputs,
                instructions=system_prompt,
                tools=tools,
                stream=False,
            )

            # file_search is handled server-side; collect client-side calls
            function_calls = [o for o in response.output if o.type == "function_call"]
            if not function_calls:
                return response.output_text  # Done — no more tool calls

            # Execute each function call and feed results back
            inputs = []
            for fc in function_calls:
                result = self._tools[fc.name](**json.loads(fc.arguments))
                inputs.append(fc)
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": result,
                    }
                )
```

In a typical query, the agent first calls `file_search` (handled server-side). If the retrieved context is insufficient — say, a question about a recent outage whose postmortem hasn't been indexed yet — the agent calls `list_local_files` to discover available documents, `read_local_file` to inspect the relevant one, and `index_document` to add it to the vector store. Then it searches again with the enriched store and writes its final answer.

The `index_document` tool is worth highlighting — it's the agent actively curating its own knowledge base:

```python
class ResearchAgent:
    ...

    def _index_document(self, file_path):
        """Upload a local file to the vector store so it becomes searchable."""
        file = self.client.files.create(
            file=open(file_path, "rb"), purpose="assistants"
        )
        attach = self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id, file_id=file.id
        )
        while attach.status == "in_progress":
            time.sleep(0.5)
            attach = self.client.vector_stores.files.retrieve(
                vector_store_id=self.vector_store_id, file_id=file.id
            )
        return f"Indexed {file_path} (file_id={file.id}, status={attach.status})"
```

This uses the Files API to upload the document and `vector_stores.files.create()` to attach it to the store. After polling until indexing completes, the file is searchable by `file_search` in subsequent turns of the same query — or in future queries.

The `from_files` classmethod handles initial vector store setup the same way, populating the store with a set of known documents at startup:

```python
class ResearchAgent:
    ...

    @classmethod
    def from_files(cls, client, model, name, file_paths, local_docs_dir=None):
        """Create a ResearchAgent with a vector store populated from files."""
        vector_store = client.vector_stores.create(name=name)
        for path in file_paths:
            file = client.files.create(file=open(path, "rb"), purpose="assistants")
            client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file.id
            )
            # ... poll until indexing completes ...
        return cls(client, model, vector_store.id, local_docs_dir)
```

## The Optimization Loop

The optimization loop is deliberately *not* an agent. The workflow is always the same fixed sequence: read prompt, evaluate, score, propose improvement, save. There's no decision-making that benefits from LLM tool selection — so it's just a function.

### Evaluation

`evaluate_prompt` runs the research agent on every test case and judges each answer with the judge model. It's deterministic — every test case is always evaluated, no matter what:

```python
def evaluate_prompt(client, judge_model, research_agent, system_prompt, test_cases):
    """Run the research agent on all test cases and judge each answer."""
    results = []
    for tc in test_cases:
        answer = research_agent.query(tc["question"], system_prompt)
        judgment = client.responses.create(
            model=judge_model,
            input=(
                f"Score the following answer on a scale of 0.0 to 1.0.\n\n"
                f"Question: {tc['question']}\n"
                f"Expected: {tc['expected']}\nActual: {answer}\n\n"
                f'Respond with JSON: {{"score": <float>, "reasoning": "..."}}'
            ),
            stream=False,
        )
        score_data = json.loads(judgment.output_text)
        results.append({**tc, "actual": answer, **score_data})

    avg_score = sum(r["score"] for r in results) / len(results)
    return {"results": results, "average_score": avg_score}
```

### Prompt proposal

`propose_new_prompt` takes the current prompt and judge feedback, and uses the judge model to generate an improved version. The judge does double duty — scoring answers *and* proposing improvements based on its own feedback:

```python
def propose_new_prompt(client, judge_model, current_prompt, feedback):
    """Use the judge model to generate an improved system prompt."""
    response = client.responses.create(
        model=judge_model,
        input=(
            f"Improve this research agent's system prompt based on feedback.\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Return ONLY the improved prompt text."
        ),
        stream=False,
    )
    return response.output_text.strip()
```

### The loop

`optimize_prompt` ties it all together in a plain `for` loop:

```python
def optimize_prompt(
    client, judge_model, research_agent, ledger, prompt_id, test_cases, max_iterations=5
):
    """Deterministic optimization loop — no LLM-driven tool selection."""
    for iteration in range(max_iterations):
        # 1. Read current prompt
        current = client.prompts.retrieve(prompt_id)

        # 2. Run research agent on all test cases and judge answers
        eval_result = evaluate_prompt(
            client, judge_model, research_agent, current.prompt, test_cases
        )

        # 3. Log scores to ledger
        feedback_summary = "; ".join(
            f"Q: {r['question'][:40]}… → {r['score']:.1f} ({r['reasoning']})"
            for r in eval_result["results"]
        )
        ledger.log(
            prompt_id, current.version, eval_result["average_score"], feedback_summary
        )

        # 4. Propose improved prompt using judge model
        new_prompt = propose_new_prompt(
            client, judge_model, current.prompt, feedback_summary
        )

        # 5. Save new version via Prompts API
        client.prompts.update(prompt_id, prompt=new_prompt, version=current.version)
```

This is orchestration code, not an agent. Each step is a plain function call. The Prompts API auto-increments versions on each `update()`, and the `version` parameter provides optimistic locking so concurrent experiments don't silently overwrite each other.

## Running It

First, make sure you have a Llama Stack server running with Ollama:

```bash
ollama pull llama3.1:8b
ollama pull gpt-oss:20b
uv run --with llama-stack llama stack run ollama
```

Then set up the research agent with some engineering documents. Some docs are indexed in the vector store up front; others live in a local directory for the agent to discover and index on demand:

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")
MODEL = "ollama/llama3.1:8b"

# Some docs are indexed up front; others are in a local directory
# for the research agent to discover and index on demand.
research_agent = ResearchAgent.from_files(
    client,
    model=MODEL,
    name="engineering-kb",
    file_paths=[
        "docs/blog/sample_docs/design/user_service_v2.md",
        "docs/blog/sample_docs/runbooks/deployment_rollback.md",
    ],
    local_docs_dir="docs/blog/sample_docs/postmortems",
)

# Verify the research agent works
answer = research_agent.query(
    question="What is the deployment rollback procedure?",
    system_prompt="Answer based on the provided context.",
)
print(f"Research agent says: {answer}")

# Create the initial system prompt via Prompts API
initial = client.prompts.create(
    prompt="You are a helpful assistant. Answer questions based on the provided context.",
)

# Run the deterministic optimization loop
optimize_prompt(
    client=client,
    judge_model="ollama/gpt-oss:20b",
    research_agent=research_agent,
    ledger=ScoreLedger(),
    prompt_id=initial.prompt_id,
    test_cases=[
        {
            "question": "What is the deployment rollback procedure?",
            "expected": "Revert the Kubernetes deployment to the previous revision "
            "using kubectl rollout undo",
        },
        {
            "question": "What authentication method does the user service use?",
            "expected": "JWT tokens issued by the auth gateway with RS256 signing",
        },
        {
            "question": "What was the root cause of the 2025-02 checkout outage?",
            "expected": "Connection pool exhaustion in the payments service "
            "due to missing timeout configuration",
        },
    ],
    max_iterations=5,
)

# Show the best prompt
result = best_prompt(client, ScoreLedger(), initial.prompt_id)
print(f"Best prompt (v{result['version']}, score={result['score']:.2f}):")
print(f"  {result['prompt']}")
```

The full implementation with tool schema generation and all supporting code is available at [prompt_optimizer_sketch.py](./prompt_optimizer_sketch.py).

## How It Works Under the Hood

The two-tier architecture cleanly separates the agentic and deterministic parts of the system:

**The research agent** uses *both* kinds of Responses API tools:

- **Server-side tools** like `file_search` are executed automatically — the Responses API searches the vector store, retrieves relevant chunks, and feeds them to the model without any client code. This is what makes knowledge base search a single API call.
- **Client-side function tools** (`read_local_file`, `index_document`, `list_local_files`) return tool call objects for you to execute. The `while True` loop dispatches these, and the results feed back into the next `responses.create()` call. This is what lets the agent actively curate its knowledge base.

The agent combines both in a single loop: `file_search` results come back automatically within the response, while function calls need client-side execution. The model sees both sources of information and decides what to do next.

**The optimization loop** doesn't need any of this machinery. It calls `responses.create()` directly for judging and prompt generation — no tool calling, no agentic loop, just straightforward LLM calls. The Prompts API stores versioned prompt text with optimistic locking, and the SQLite ledger tracks how well each version performed.

## What's Next

The pattern here — an agentic research agent inside a deterministic optimization loop — generalizes well beyond prompt tuning:

- **MCP tools** for connecting to external services (databases, APIs, code execution sandboxes) — the research agent could pull in live data alongside static documents
- **Web search** alongside `file_search` for agents that combine local knowledge with live web results
- **Multiple research agents** with different vector stores, optimized in parallel — each specializing in a different knowledge domain

To learn more:
- [Responses API documentation](/docs/building_applications/responses_vs_agents)
- [Conversations API documentation](/docs/api-openai/conformance#conversations)
- [OpenAI API compatibility](/docs/api-openai)
- [Vector Stores documentation](/docs/building_applications/rag)
- [Join our Discord](https://discord.gg/llama-stack)
