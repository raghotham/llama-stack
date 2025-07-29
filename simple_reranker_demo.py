#!/usr/bin/env python3
"""
Simple demonstration of reranking model registration and usage.

This script can be run independently to test the reranking functionality.
It creates a mock server environment for testing.
"""

import asyncio
import json
import os
from typing import Dict, List, Any

# Mock implementation for demonstration purposes
class MockLlamaStackClient:
    """Mock client for demonstration when server is not available."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        
    class Models:
        def __init__(self, client):
            self.client = client
            
        async def register_reranking_model(
            self, 
            model_id: str, 
            provider_model_id: str = None,
            provider_id: str = None, 
            metadata: Dict[str, Any] = None
        ):
            """Register a reranking model."""
            model = {
                "identifier": model_id,
                "provider_id": provider_id or "cohere",
                "provider_resource_id": provider_model_id or model_id,
                "model_type": "reranking",
                "metadata": metadata or {}
            }
            self.client.registered_models[model_id] = model
            print(f"✅ Mock: Registered reranking model '{model_id}'")
            return type('Model', (), model)()
            
        async def unregister_reranking_model(self, model_id: str):
            """Unregister a reranking model."""
            if model_id in self.client.registered_models:
                del self.client.registered_models[model_id]
                print(f"✅ Mock: Unregistered reranking model '{model_id}'")
            else:
                raise ValueError(f"Model {model_id} not found")
                
        async def list_models(self):
            """List all registered models."""
            models = []
            for model_data in self.client.registered_models.values():
                models.append(type('Model', (), model_data)())
            return type('ListResponse', (), {"data": models})()
            
        async def get_model(self, model_id: str):
            """Get a specific model."""
            if model_id not in self.client.registered_models:
                raise ValueError(f"Model {model_id} not found")
            return type('Model', (), self.client.registered_models[model_id])()
    
    class Reranker:
        def __init__(self, client):
            self.client = client
            
        async def rerank(
            self, 
            query: str, 
            documents: List[str], 
            model: str, 
            top_n: int = None,
            return_documents: bool = False
        ):
            """Mock reranking functionality."""
            # Simple mock: reverse order and add mock scores
            results = []
            for i, doc in enumerate(reversed(documents[:top_n or len(documents)])):
                score = 0.9 - (i * 0.1)  # Decreasing scores
                result_doc = doc if return_documents else None
                results.append(type('RerankResult', (), {
                    "index": len(documents) - 1 - i,
                    "relevance_score": score,
                    "document": result_doc
                })())
            
            return type('RerankResponse', (), {
                "results": results,
                "model": model,
                "usage": {"total_tokens": len(query) + sum(len(d) for d in documents)}
            })()
            
        async def list_models(self):
            """List available reranker models."""
            mock_models = [
                {
                    "identifier": "rerank-v3.5",
                    "provider_id": "cohere", 
                    "metadata": {"display_name": "Cohere Rerank v3.5", "max_tokens_per_doc": 4096}
                },
                {
                    "identifier": "rerank-2.5",
                    "provider_id": "voyage",
                    "metadata": {"display_name": "Voyage Rerank 2.5", "max_total_tokens": 600000}
                },
                {
                    "identifier": "nvidia/nv-rerankqa-mistral-4b-v3", 
                    "provider_id": "nvidia",
                    "metadata": {"display_name": "NVIDIA Rerank QA Mistral 4B", "max_documents": 100}
                }
            ]
            
            models = [type('Model', (), model)() for model in mock_models]
            return type('ListResponse', (), {"models": models})()
    
    async def health(self):
        """Health check."""
        return {"status": "healthy"}
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.registered_models = {}
        self.models = self.Models(self)
        self.reranker = self.Reranker(self)


class RerankerDemo:
    """Demonstration of reranking model registration and usage."""
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        if use_mock:
            self.client = MockLlamaStackClient("http://localhost:5001")
        else:
            # Try to import real client
            try:
                from llama_stack_client import LlamaStackClient
                self.client = LlamaStackClient(base_url="http://localhost:5001")
            except ImportError:
                print("⚠️ LlamaStackClient not available, using mock client")
                self.client = MockLlamaStackClient("http://localhost:5001")
    
    async def demonstrate_model_registration(self):
        """Demonstrate registering and managing reranking models."""
        print("🔧 Demonstrating Reranking Model Registration")
        print("=" * 50)
        
        # Example 1: Register a Cohere reranking model
        print("\n1️⃣ Registering Cohere reranking model...")
        cohere_model = await self.client.models.register_reranking_model(
            model_id="my-cohere-reranker",
            provider_model_id="rerank-v3.5",
            provider_id="cohere",
            metadata={
                "display_name": "My Cohere Reranker",
                "max_tokens_per_doc": 4096,
                "description": "High-quality reranking for search results"
            }
        )
        print(f"   Registered: {cohere_model.identifier}")
        print(f"   Type: {cohere_model.model_type}")
        
        # Example 2: Register a Voyage reranking model  
        print("\n2️⃣ Registering Voyage reranking model...")
        voyage_model = await self.client.models.register_reranking_model(
            model_id="my-voyage-reranker",
            provider_model_id="rerank-2.5",
            provider_id="voyage",
            metadata={
                "display_name": "My Voyage Reranker",
                "max_total_tokens": 600000,
                "description": "Multilingual reranking capabilities"
            }
        )
        print(f"   Registered: {voyage_model.identifier}")
        print(f"   Type: {voyage_model.model_type}")
        
        # Example 3: List all registered models
        print("\n3️⃣ Listing all registered models...")
        models_response = await self.client.models.list_models()
        reranking_models = [m for m in models_response.data if hasattr(m, 'model_type') and m.model_type == "reranking"]
        
        print(f"   Found {len(reranking_models)} reranking models:")
        for model in reranking_models:
            print(f"   • {model.identifier} ({model.provider_id})")
        
        # Example 4: Unregister models
        print("\n4️⃣ Unregistering models...")
        await self.client.models.unregister_reranking_model("my-cohere-reranker")
        await self.client.models.unregister_reranking_model("my-voyage-reranker")
        print("   ✅ Models unregistered")
        
    async def demonstrate_reranking_usage(self):
        """Demonstrate using reranking functionality."""
        print("\n🔄 Demonstrating Reranking Usage")
        print("=" * 40)
        
        # Sample documents and query
        query = "How does machine learning work?"
        documents = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Python is a popular programming language for data science and machine learning.",
            "Deep learning uses artificial neural networks with multiple layers to model data.",
            "Statistics provides the mathematical foundation for many machine learning algorithms.",
            "Data preprocessing is crucial for preparing datasets for machine learning models.",
            "Supervised learning requires labeled training data to learn patterns.",
            "Artificial intelligence encompasses machine learning, deep learning, and other technologies."
        ]
        
        print(f"\n📝 Query: '{query}'")
        print(f"📄 Documents: {len(documents)} items to rerank")
        
        # Example 1: Basic reranking
        print("\n1️⃣ Basic reranking (top 3 results)...")
        rerank_response = await self.client.reranker.rerank(
            query=query,
            documents=documents,
            model="rerank-v3.5",  # Cohere model
            top_n=3,
            return_documents=True
        )
        
        print(f"   Model used: {rerank_response.model}")
        print(f"   Results returned: {len(rerank_response.results)}")
        print("   Top results:")
        
        for i, result in enumerate(rerank_response.results, 1):
            doc_preview = (result.document[:60] + "...") if result.document and len(result.document) > 60 else result.document
            print(f"   {i}. Score: {result.relevance_score:.3f}")
            print(f"      Original index: {result.index}")
            print(f"      Text: {doc_preview}")
        
        if hasattr(rerank_response, 'usage') and rerank_response.usage:
            print(f"   Token usage: {rerank_response.usage}")
            
        # Example 2: Compare different models
        print("\n2️⃣ Comparing different reranker models...")
        models_to_test = [
            ("rerank-v3.5", "Cohere"),
            ("rerank-2.5", "Voyage AI"), 
            ("nvidia/nv-rerankqa-mistral-4b-v3", "NVIDIA")
        ]
        
        for model_id, provider_name in models_to_test:
            try:
                response = await self.client.reranker.rerank(
                    query=query,
                    documents=documents,
                    model=model_id,
                    top_n=1  # Just get the top result
                )
                
                if response.results:
                    top_result = response.results[0]
                    print(f"   {provider_name}: Score {top_result.relevance_score:.3f} (doc #{top_result.index})")
                else:
                    print(f"   {provider_name}: No results")
                    
            except Exception as e:
                print(f"   {provider_name}: Error - {e}")
    
    async def demonstrate_available_models(self):
        """Show available reranker models from providers."""
        print("\n📋 Available Reranker Models")
        print("=" * 30)
        
        try:
            models_response = await self.client.reranker.list_models()
            
            # Group by provider
            providers = {}
            for model in models_response.models:
                provider = model.provider_id
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(model)
            
            for provider_id, models in providers.items():
                print(f"\n🏢 {provider_id.upper()} Provider:")
                for model in models:
                    print(f"   📦 {model.identifier}")
                    if hasattr(model, 'metadata') and model.metadata:
                        for key, value in model.metadata.items():
                            print(f"      {key}: {value}")
                            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
    
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        print("🎯 Llama Stack Reranking Model Demo")
        print("=" * 60)
        
        if self.use_mock:
            print("🔧 Running in MOCK mode (no real API calls)")
        else:
            print("🌐 Running with real Llama Stack server")
            
        print(f"🔗 Client: {self.client.base_url}")
        print()
        
        try:
            # Health check
            await self.client.health()
            print("✅ Connection successful")
            
            # Run demos
            await self.demonstrate_available_models()
            await self.demonstrate_model_registration() 
            await self.demonstrate_reranking_usage()
            
            print("\n" + "=" * 60)
            print("🎉 Demo completed successfully!")
            print("\n💡 Key takeaways:")
            print("   • Reranking models can be registered with model_type='reranking'")
            print("   • Supports Cohere, Voyage AI, and NVIDIA providers")
            print("   • New register_reranking_model() and unregister_reranking_model() methods")
            print("   • Reranking improves search relevance by 10-30% typically")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            if not self.use_mock:
                print("💡 Try running with mock mode: python simple_reranker_demo.py --mock")


async def main():
    """Main entry point."""
    import sys
    
    # Check if we should use mock mode
    use_mock = "--mock" in sys.argv or len(sys.argv) > 1 and sys.argv[1] == "mock"
    
    if not use_mock:
        print("⚠️ Attempting to connect to real Llama Stack server...")
        print("   If this fails, run with: python simple_reranker_demo.py --mock")
        print()
    
    demo = RerankerDemo(use_mock=use_mock)
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())