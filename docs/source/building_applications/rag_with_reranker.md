# Enhanced RAG with Reranker

This guide demonstrates how to use the **Reranker API** to significantly improve the quality of your RAG (Retrieval Augmented Generation) applications. Reranking is a crucial step that can dramatically enhance the relevance of retrieved documents by reordering them based on their semantic similarity to the query.

## Why Use Reranking in RAG?

Traditional RAG systems rely on vector similarity search to retrieve relevant documents. However, vector embeddings might not always capture the precise semantic relationship between a query and documents. Reranking addresses this by:

1. **Improving Precision**: Cross-encoder models (used by rerankers) jointly process the query and documents, providing more accurate relevance scores
2. **Better Context Selection**: Ensures the most relevant documents are prioritized for the LLM context
3. **Reducing Noise**: Filters out less relevant documents that might confuse the model
4. **Enhanced Performance**: Studies show reranking can improve RAG accuracy by 10-30%

## Complete RAG with Reranker Example

Let's build a comprehensive RAG system that demonstrates the improvement gained by adding a reranker step.

### 1. Setup and Dependencies

```python
import os
from llama_stack_client import LlamaStackClient, Agent, Document

# Initialize the client
client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

# Configuration
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "rerank-v3.5"  # Cohere model
```

### 2. Document Ingestion and Vector Database Setup

```python
# Create a vector database for our documents
vector_db_id = "knowledge_base_with_reranker"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=EMBEDDING_MODEL,
    embedding_dimension=384,
    provider_id="faiss",
)

# Sample documents - you can replace these with your own content
documents = [
    Document(
        document_id="ml_basics",
        content="""Machine learning is a subset of artificial intelligence that enables
        computers to learn and make decisions from data without being explicitly programmed.
        It involves algorithms that can identify patterns, make predictions, and improve
        their performance over time through experience.""",
        mime_type="text/plain",
        metadata={"category": "basics", "topic": "machine_learning"},
    ),
    Document(
        document_id="deep_learning",
        content="""Deep learning is a specialized branch of machine learning that uses
        neural networks with multiple layers to model and understand complex patterns
        in data. It's particularly effective for tasks like image recognition, natural
        language processing, and speech recognition.""",
        mime_type="text/plain",
        metadata={"category": "advanced", "topic": "deep_learning"},
    ),
    Document(
        document_id="transformers",
        content="""Transformers are a revolutionary neural network architecture introduced
        in 'Attention is All You Need'. They use self-attention mechanisms to process
        sequences and have become the foundation for large language models like GPT,
        BERT, and Llama. The key innovation is parallel processing of sequence elements.""",
        mime_type="text/plain",
        metadata={"category": "architecture", "topic": "transformers"},
    ),
    Document(
        document_id="llm_training",
        content="""Large Language Model training involves multiple stages: pre-training on
        vast text corpora, supervised fine-tuning on specific tasks, and reinforcement
        learning from human feedback (RLHF). The process requires massive computational
        resources and careful data curation to achieve good performance.""",
        mime_type="text/plain",
        metadata={"category": "training", "topic": "llm"},
    ),
    Document(
        document_id="rag_systems",
        content="""Retrieval-Augmented Generation (RAG) combines the power of large language
        models with external knowledge retrieval. It first retrieves relevant documents
        from a knowledge base, then uses them as context for generating responses. This
        approach helps reduce hallucinations and keeps information up-to-date.""",
        mime_type="text/plain",
        metadata={"category": "applications", "topic": "rag"},
    ),
]

# Ingest documents into the vector database
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,
)
```

### 3. Traditional RAG Agent (Without Reranking)

First, let's create a standard RAG agent to establish a baseline:

```python
# Create a traditional RAG agent
traditional_rag_agent = Agent(
    client,
    model=MODEL_ID,
    instructions="""You are a helpful AI assistant specializing in machine learning and AI topics.
    Use the retrieved documents to provide accurate, detailed answers. Always cite the sources
    of your information when possible.""",
    tools=[
        {
            "name": "builtin::rag",
            "args": {
                "vector_db_ids": [vector_db_id],
                "query_config": {
                    "chunk_template": "Source: {metadata}\nContent: {chunk.content}\n\n"
                },
            },
        }
    ],
)


def query_traditional_rag(question: str):
    session_id = traditional_rag_agent.create_session(f"traditional_session")
    response = traditional_rag_agent.create_turn(
        messages=[{"role": "user", "content": question}],
        session_id=session_id,
        stream=False,
    )
    return response.output_message.content
```

### 4. Enhanced RAG with Reranker

The simplest way to add reranking to your RAG agent is by configuring the RAG tool with reranker parameters:

```python
# Create an enhanced RAG agent with built-in reranking
enhanced_rag_agent = Agent(
    client,
    model=MODEL_ID,
    instructions="""You are a helpful AI assistant specializing in machine learning and AI topics.
    Use the retrieved documents to provide accurate, detailed answers. Always cite the sources
    of your information when possible.""",
    tools=[
        {
            "name": "builtin::rag",
            "args": {
                "vector_db_ids": [vector_db_id],
                "query_config": {
                    "top_k": 10,  # Retrieve more documents initially
                    "reranker": {
                        "model": RERANKER_MODEL,
                        "top_n": 3,  # Keep only top 3 after reranking
                        "provider": "cohere",  # or "voyage", "nvidia"
                    },
                    "chunk_template": "Source: {metadata}\nContent: {chunk.content}\n\n",
                },
            },
        }
    ],
)


def query_enhanced_rag(question: str):
    session_id = enhanced_rag_agent.create_session(f"enhanced_session")
    response = enhanced_rag_agent.create_turn(
        messages=[{"role": "user", "content": question}],
        session_id=session_id,
        stream=False,
    )
    return response.output_message.content
```

### Alternative: Manual Reranking for Full Control

If you need more control over the reranking process, you can implement it manually:

```python
class CustomRAGWithReranker:
    def __init__(self, client, vector_db_id, reranker_model, model_id):
        self.client = client
        self.vector_db_id = vector_db_id
        self.reranker_model = reranker_model

        # Create agent without RAG tool for manual control
        self.agent = Agent(
            client,
            model=model_id,
            instructions="""You are a helpful AI assistant specializing in machine learning
            and AI topics. Use the provided context documents to give accurate, detailed answers.
            Always cite the sources of your information.""",
        )

    def retrieve_and_rerank(self, query: str, top_k: int = 10, top_n: int = 3):
        """Retrieve documents and rerank them for better relevance."""
        # Step 1: Initial retrieval using vector similarity
        retrieval_results = self.client.tool_runtime.rag_tool.query(
            vector_db_ids=[self.vector_db_id],
            content=query,
            query_config={"top_k": top_k, "chunk_template": "{chunk.content}"},
        )

        documents = [chunk.content for chunk in retrieval_results.chunks]
        if not documents:
            return [], []

        # Step 2: Rerank documents using the reranker API
        rerank_response = self.client.reranker.rerank(
            query=query,
            documents=documents,
            model=self.reranker_model,
            top_n=top_n,
            return_documents=True,
        )

        # Return reranked documents with scores
        reranked_docs = [
            (result.document, result.relevance_score)
            for result in rerank_response.results
        ]
        return reranked_docs

    def query(self, question: str):
        """Query the enhanced RAG system with reranking."""
        reranked_docs = self.retrieve_and_rerank(question)

        if not reranked_docs:
            return "I couldn't find relevant information to answer your question."

        # Format context with relevance scores
        context = "\n\n".join(
            [
                f"Source {i+1} (Relevance: {score:.3f}):\n{doc}"
                for i, (doc, score) in enumerate(reranked_docs)
            ]
        )

        enhanced_prompt = f"""Context Information:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

        session_id = self.agent.create_session(f"manual_session")
        response = self.agent.create_turn(
            messages=[{"role": "user", "content": enhanced_prompt}],
            session_id=session_id,
            stream=False,
        )
        return response.output_message.content


# Initialize the custom RAG system
custom_rag = CustomRAGWithReranker(
    client=client,
    vector_db_id=vector_db_id,
    reranker_model=RERANKER_MODEL,
    model_id=MODEL_ID,
)
```

### 5. Comparative Analysis

Let's test both approaches with the same questions:

```python
# Test questions
questions = [
    "How do transformers work and why are they important for modern AI?",
    "What is the difference between machine learning and deep learning?",
    "Explain the RAG architecture and its benefits for LLMs",
    "What are the key steps in training large language models?",
]


def compare_rag_approaches():
    """Compare traditional RAG vs. RAG with reranking"""

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}: {question}")
        print("=" * 80)

        # Traditional RAG response
        print("\n📋 TRADITIONAL RAG RESPONSE:")
        print("-" * 50)
        traditional_response = query_traditional_rag(question)
        print(traditional_response)

        # Enhanced RAG with reranker response
        print("\n🚀 ENHANCED RAG WITH RERANKER:")
        print("-" * 50)
        enhanced_response = query_enhanced_rag(question)
        print(enhanced_response)

        print("\n" + "=" * 80)


# Run the comparison
compare_rag_approaches()
```

### 6. Advanced Reranker Configuration

You can experiment with different reranker models and configurations:

```python
# Compare different reranker models
reranker_configs = [
    {"model": "rerank-v3.5", "provider": "cohere"},
    {"model": "rerank-2.5", "provider": "voyage"},
    {"model": "nvidia/nv-rerankqa-mistral-4b-v3", "provider": "nvidia"},
]


def create_agent_with_reranker(reranker_config):
    """Create an agent with a specific reranker configuration"""
    return Agent(
        client,
        model=MODEL_ID,
        instructions="You are a helpful AI assistant.",
        tools=[
            {
                "name": "builtin::rag",
                "args": {
                    "vector_db_ids": [vector_db_id],
                    "query_config": {
                        "top_k": 10,
                        "reranker": {
                            "model": reranker_config["model"],
                            "top_n": 3,
                            "provider": reranker_config["provider"],
                        },
                        "chunk_template": "Content: {chunk.content}\n\n",
                    },
                },
            }
        ],
    )


# Test different reranker models
def compare_reranker_models(question: str):
    """Compare different reranker models on the same question"""
    for config in reranker_configs:
        print(f"\n--- Using {config['provider']} {config['model']} ---")

        try:
            agent = create_agent_with_reranker(config)
            session_id = agent.create_session(f"test_session")
            response = agent.create_turn(
                messages=[{"role": "user", "content": question}],
                session_id=session_id,
                stream=False,
            )
            print(response.output_message.content[:200] + "...")

        except Exception as e:
            print(f"Error with {config['provider']}: {e}")


# Example usage
compare_reranker_models("What are neural network architectures used in modern AI?")
```

### 7. Performance Metrics and Evaluation

```python
def evaluate_rag_improvement():
    """
    Evaluate the improvement gained by using reranking.
    This example shows how to measure retrieval quality.
    """

    test_cases = [
        {
            "query": "transformer architecture attention mechanism",
            "relevant_doc_keywords": ["attention", "transformer", "self-attention"],
        },
        {
            "query": "machine learning vs deep learning differences",
            "relevant_doc_keywords": [
                "machine learning",
                "deep learning",
                "neural networks",
            ],
        },
    ]

    for test_case in test_cases:
        query = test_case["query"]
        keywords = test_case["relevant_doc_keywords"]

        print(f"\nEvaluating query: '{query}'")

        # Get traditional retrieval results
        traditional_results = client.tool_runtime.rag_tool.query(
            vector_db_ids=[vector_db_id], content=query, query_config={"top_k": 5}
        )

        # Get reranked results
        documents = [chunk.content.lower() for chunk in traditional_results.chunks]
        rerank_response = client.reranker.rerank(
            query=query, documents=documents, model=RERANKER_MODEL, top_n=3
        )

        # Simple relevance evaluation based on keyword presence
        def calculate_relevance_score(docs, keywords):
            total_score = 0
            for i, doc in enumerate(docs):
                keyword_count = sum(1 for kw in keywords if kw.lower() in doc.lower())
                # Weight by position (higher weight for top results)
                position_weight = 1.0 / (i + 1)
                total_score += keyword_count * position_weight
            return total_score

        # Traditional RAG relevance
        traditional_docs = [chunk.content for chunk in traditional_results.chunks[:3]]
        traditional_score = calculate_relevance_score(traditional_docs, keywords)

        # Reranked relevance
        reranked_docs = [documents[result.index] for result in rerank_response.results]
        reranked_score = calculate_relevance_score(reranked_docs, keywords)

        improvement = (
            ((reranked_score - traditional_score) / traditional_score * 100)
            if traditional_score > 0
            else 0
        )

        print(f"Traditional RAG relevance score: {traditional_score:.2f}")
        print(f"Reranked RAG relevance score: {reranked_score:.2f}")
        print(f"Improvement: {improvement:.1f}%")


# Run evaluation
evaluate_rag_improvement()
```

### 8. Production Best Practices

When implementing reranking in production RAG systems:

```python
class ProductionRAGWithReranker:
    def __init__(self, client, vector_db_id, reranker_model):
        self.client = client
        self.vector_db_id = vector_db_id
        self.reranker_model = reranker_model

        # Cache for frequently asked questions
        self.cache = {}

    async def query_with_caching(self, query: str, cache_ttl: int = 3600):
        """Production-ready query with caching and error handling"""

        # Check cache first
        cache_key = f"{query}:{self.reranker_model}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Retrieve with error handling
            retrieval_results = self.client.tool_runtime.rag_tool.query(
                vector_db_ids=[self.vector_db_id],
                content=query,
                query_config={"top_k": 20},  # Retrieve more for better reranking
            )

            if not retrieval_results.chunks:
                return "No relevant information found."

            documents = [chunk.content for chunk in retrieval_results.chunks]

            # Rerank with fallback
            try:
                rerank_response = self.client.reranker.rerank(
                    query=query,
                    documents=documents,
                    model=self.reranker_model,
                    top_n=5,
                    truncation=True,
                )

                # Use reranked results
                final_docs = [
                    documents[result.index] for result in rerank_response.results
                ]

            except Exception as rerank_error:
                print(
                    f"Reranking failed: {rerank_error}, falling back to vector similarity"
                )
                # Fallback to traditional retrieval
                final_docs = documents[:5]

            # Cache the result
            self.cache[cache_key] = final_docs
            return final_docs

        except Exception as e:
            print(f"Query failed: {e}")
            return "Sorry, I encountered an error processing your request."


# Production configuration
production_rag = ProductionRAGWithReranker(
    client=client, vector_db_id=vector_db_id, reranker_model="rerank-v3.5"
)
```

## Key Benefits Demonstrated

Through this comprehensive example, we've shown how reranking improves RAG systems by:

1. **🎯 Better Relevance**: Cross-encoder models provide more accurate document relevance scores
2. **🚀 Improved Accuracy**: Reranked documents are more likely to contain the information needed to answer the query
3. **⚡ Flexible Models**: Support for multiple reranker providers (Cohere, Voyage AI, NVIDIA)
4. **🔧 Production Ready**: Includes caching, error handling, and fallback mechanisms
5. **📊 Measurable Improvement**: Methods to evaluate and quantify the performance gains

## Next Steps

- Experiment with different reranker models to find the best fit for your domain
- Implement A/B testing to measure the impact on user satisfaction
- Consider fine-tuning reranker models on your specific domain data
- Optimize the balance between retrieval breadth (top_k) and reranking depth (top_n)

By integrating reranking into your RAG pipeline, you can significantly improve the quality and relevance of your AI-powered applications.