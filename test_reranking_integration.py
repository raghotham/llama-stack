#!/usr/bin/env python3
"""
Integration test script for reranking model registration and usage.

This script demonstrates:
1. Registering a new reranking model
2. Using the reranking model for document reranking  
3. Unregistering the reranking model

Prerequisites:
- Llama Stack server running with reranker providers configured
- API keys for reranker providers (Cohere, Voyage AI, or NVIDIA)
"""

import asyncio
import os
import sys
from typing import Dict, Any

from llama_stack_client import LlamaStackClient


class RerankerIntegrationTest:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.client = LlamaStackClient(base_url=base_url)
        self.test_model_id = "test-reranker-model"
        
    async def test_model_registration(self) -> bool:
        """Test registering and unregistering a reranking model."""
        print("🧪 Testing reranking model registration...")
        
        try:
            # Test 1: Register a reranking model
            print("\n1️⃣ Registering reranking model...")
            registered_model = await self.client.models.register_reranking_model(
                model_id=self.test_model_id,
                provider_model_id="rerank-v3.5",  # Cohere model
                provider_id="cohere",
                metadata={
                    "display_name": "Test Cohere Rerank Model",
                    "max_tokens_per_doc": 4096,
                    "test_model": True
                }
            )
            
            print(f"✅ Successfully registered: {registered_model.identifier}")
            print(f"   Model type: {registered_model.model_type}")
            print(f"   Provider: {registered_model.provider_id}")
            print(f"   Metadata: {registered_model.metadata}")
            
            # Test 2: Verify the model is listed
            print("\n2️⃣ Verifying model appears in models list...")
            models = await self.client.models.list_models()
            
            registered_models = [m for m in models.data if m.identifier == self.test_model_id]
            if not registered_models:
                print("❌ Model not found in models list")
                return False
                
            print(f"✅ Found model in list: {registered_models[0].identifier}")
            
            # Test 3: Get specific model details
            print("\n3️⃣ Getting model details...")
            model_details = await self.client.models.get_model(self.test_model_id)
            print(f"✅ Retrieved model: {model_details.identifier}")
            print(f"   Type: {model_details.model_type}")
            
            # Test 4: Unregister the model
            print("\n4️⃣ Unregistering reranking model...")
            await self.client.models.unregister_reranking_model(self.test_model_id)
            print("✅ Successfully unregistered model")
            
            # Test 5: Verify model is removed
            print("\n5️⃣ Verifying model is removed...")
            try:
                await self.client.models.get_model(self.test_model_id)
                print("❌ Model still exists after unregistration")
                return False
            except Exception:
                print("✅ Model properly removed")
                
            return True
            
        except Exception as e:
            print(f"❌ Error during model registration test: {e}")
            # Cleanup: try to unregister if it exists
            try:
                await self.client.models.unregister_reranking_model(self.test_model_id)
            except:
                pass
            return False
    
    async def test_reranking_functionality(self) -> bool:
        """Test actual reranking functionality with a provider."""
        print("\n🔄 Testing reranking functionality...")
        
        # Check if we have API keys for any provider
        providers_to_test = [
            ("cohere", "COHERE_API_KEY", "rerank-v3.5"),
            ("voyage", "VOYAGE_API_KEY", "rerank-2.5"),
            ("nvidia", "NVIDIA_API_KEY", "nvidia/nv-rerankqa-mistral-4b-v3"),
        ]
        
        working_provider = None
        for provider_id, env_key, model_name in providers_to_test:
            if os.getenv(env_key):
                working_provider = (provider_id, model_name)
                print(f"✅ Found API key for {provider_id}")
                break
        
        if not working_provider:
            print("⚠️ No API keys found for reranker providers. Skipping reranking test.")
            print("   Set COHERE_API_KEY, VOYAGE_API_KEY, or NVIDIA_API_KEY to test reranking.")
            return True
        
        provider_id, model_name = working_provider
        
        try:
            # Test documents and query
            query = "What is machine learning?"
            documents = [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "Deep learning is a subset of machine learning with networks capable of learning unsupervised.",
                "Artificial intelligence is the simulation of human intelligence in machines.",
                "Python is a high-level programming language used for general-purpose programming.",
                "Statistics is the discipline that concerns collection, organization, analysis of data.",
            ]
            
            print(f"\n🎯 Testing reranking with {provider_id} provider...")
            print(f"Query: '{query}'")
            print(f"Documents: {len(documents)} total")
            
            # Test reranking
            rerank_response = await self.client.reranker.rerank(
                query=query,
                documents=documents,
                model=model_name,
                top_n=3,
                return_documents=True
            )
            
            print(f"✅ Reranking successful!")
            print(f"   Model used: {rerank_response.model}")
            print(f"   Results returned: {len(rerank_response.results)}")
            
            # Display results
            print("\n📊 Reranking results (ordered by relevance):")
            for i, result in enumerate(rerank_response.results, 1):
                doc_preview = result.document[:60] + "..." if result.document and len(result.document) > 60 else result.document
                print(f"   {i}. Score: {result.relevance_score:.3f} | Doc: {doc_preview}")
            
            if rerank_response.usage:
                print(f"   Usage: {rerank_response.usage}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during reranking test: {e}")
            return False
    
    async def test_list_reranker_models(self) -> bool:
        """Test listing available reranker models."""
        print("\n📋 Testing reranker model listing...")
        
        try:
            # List models from reranker API
            models_response = await self.client.reranker.list_models()
            print(f"✅ Found {len(models_response.models)} reranker models")
            
            # Group by provider
            providers: Dict[str, list] = {}
            for model in models_response.models:
                provider = model.provider_id
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(model)
            
            for provider_id, models in providers.items():
                print(f"\n   {provider_id.upper()} Provider:")
                for model in models:
                    print(f"     • {model.identifier}")
                    if model.metadata:
                        for key, value in model.metadata.items():
                            print(f"       {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error listing reranker models: {e}")
            return False

    async def run_full_test_suite(self) -> bool:
        """Run the complete integration test suite."""
        print("🚀 Starting Reranking Integration Test Suite")
        print("=" * 60)
        
        # Check connection
        try:
            await self.client.health()
            print("✅ Connected to Llama Stack server")
        except Exception as e:
            print(f"❌ Failed to connect to Llama Stack server: {e}")
            print("   Make sure the server is running at http://localhost:5001")
            return False
        
        test_results = []
        
        # Run tests
        test_results.append(await self.test_list_reranker_models())
        test_results.append(await self.test_model_registration())
        test_results.append(await self.test_reranking_functionality())
        
        # Summary
        print("\n" + "=" * 60)
        print("🏁 Test Suite Results:")
        
        passed = sum(test_results)
        total = len(test_results)
        
        if passed == total:
            print(f"✅ All {total} tests passed!")
            return True
        else:
            print(f"❌ {total - passed} out of {total} tests failed")
            return False


async def main():
    """Main entry point for the integration test."""
    # Default to localhost, but allow override via environment
    base_url = os.getenv("LLAMA_STACK_URL", "http://localhost:5001")
    
    print("🧬 Llama Stack Reranking Integration Test")
    print(f"Server: {base_url}")
    print()
    
    # Check for API keys
    print("🔑 Checking for API keys...")
    api_keys = {
        "Cohere": os.getenv("COHERE_API_KEY", "Not set"),
        "Voyage AI": os.getenv("VOYAGE_API_KEY", "Not set"), 
        "NVIDIA": os.getenv("NVIDIA_API_KEY", "Not set"),
    }
    
    for provider, key_status in api_keys.items():
        status = "✅ Set" if key_status != "Not set" else "❌ Not set"
        print(f"   {provider}: {status}")
    
    print()
    
    # Run tests
    test_runner = RerankerIntegrationTest(base_url)
    success = await test_runner.run_full_test_suite()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())