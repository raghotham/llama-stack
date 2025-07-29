#!/usr/bin/env python3
"""
Test script to demonstrate reranking model registration and usage
with the running Llama Stack server.
"""

import asyncio
import os
from llama_stack_client import LlamaStackClient


async def test_basic_api():
    """Test basic API connectivity."""
    print("🔌 Testing basic API connectivity...")
    
    client = LlamaStackClient(base_url="http://localhost:8321")
    
    try:
        # Test health endpoint
        health = await client.health()
        print("✅ Server is healthy!")
        
        # List available models
        print("\n📋 Available models:")
        models_response = await client.models.list_models()
        for model in models_response.data:
            print(f"  • {model.identifier} (type: {model.model_type})")
        
        return True
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False


async def test_reranking_registration():
    """Test registering and using reranking models."""
    print("\n🔧 Testing reranking model registration...")
    
    client = LlamaStackClient(base_url="http://localhost:8321")
    
    try:
        # Register a reranking model
        print("\n1️⃣ Registering a reranking model...")
        
        # Check if we have API keys
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            print("⚠️ COHERE_API_KEY not set. Using fake key for demo.")
            cohere_key = "demo_key"
        
        reranking_model = await client.models.register_reranking_model(
            model_id="demo-reranker",
            provider_model_id="rerank-v3.5",
            provider_id="cohere",
            metadata={
                "display_name": "Demo Cohere Reranker",
                "max_tokens_per_doc": 4096,
                "description": "Test reranking model"
            }
        )
        
        print(f"✅ Registered reranking model: {reranking_model.identifier}")
        print(f"   Type: {reranking_model.model_type}")
        print(f"   Provider: {reranking_model.provider_id}")
        
        # List models to verify
        print("\n2️⃣ Verifying model registration...")
        models = await client.models.list_models()
        reranking_models = [m for m in models.data if m.model_type == "reranking"]
        
        print(f"✅ Found {len(reranking_models)} reranking model(s):")
        for model in reranking_models:
            print(f"   • {model.identifier}")
        
        # Get specific model
        print("\n3️⃣ Getting model details...")
        model_details = await client.models.get_model("demo-reranker")
        print(f"✅ Model details retrieved:")
        print(f"   ID: {model_details.identifier}")
        print(f"   Type: {model_details.model_type}")
        print(f"   Metadata: {model_details.metadata}")
        
        # Try to use the reranker (if API key is available)
        if cohere_key != "demo_key":
            print("\n4️⃣ Testing reranking functionality...")
            try:
                query = "What is machine learning?"
                documents = [
                    "Machine learning is a subset of artificial intelligence.",
                    "Python is a programming language.",
                    "Deep learning uses neural networks.",
                ]
                
                rerank_response = await client.reranker.rerank(
                    query=query,
                    documents=documents,
                    model="demo-reranker",
                    top_n=2,
                    return_documents=True
                )
                
                print("✅ Reranking successful!")
                for i, result in enumerate(rerank_response.results, 1):
                    print(f"   {i}. Score: {result.relevance_score:.3f} - Doc #{result.index}")
                    
            except Exception as e:
                print(f"⚠️ Reranking failed (expected without real API key): {e}")
        
        # Unregister the model
        print("\n5️⃣ Cleaning up - unregistering model...")
        await client.models.unregister_reranking_model("demo-reranker")
        print("✅ Model unregistered")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        # Try to clean up
        try:
            await client.models.unregister_reranking_model("demo-reranker")
        except:
            pass
        return False


async def test_simple_responses():
    """Test simple responses API."""
    print("\n💬 Testing simple responses API...")
    
    client = LlamaStackClient(base_url="http://localhost:8321")
    
    try:
        # Check available models for inference
        print("\n🔍 Checking available LLM models...")
        models = await client.models.list_models()
        llm_models = [m for m in models.data if m.model_type == "llm"]
        
        if not llm_models:
            print("⚠️ No LLM models available. Registering one...")
            
            # Register an Ollama model
            ollama_model = await client.models.register_model(
                model_id="llama3.2",
                provider_id="ollama",
                model_type="llm",
                metadata={}
            )
            print(f"✅ Registered model: {ollama_model.identifier}")
            llm_models = [ollama_model]
        
        model_id = llm_models[0].identifier
        print(f"📌 Using model: {model_id}")
        
        # Test chat completion
        print("\n🤖 Testing chat completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Reply in one word."}
        ]
        
        response = await client.inference.chat_completion(
            model_id=model_id,
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"✅ Response: {response.choices[0].message.content}")
        
        # Test with streaming
        print("\n🌊 Testing streaming chat completion...")
        messages = [
            {"role": "user", "content": "Count from 1 to 5."}
        ]
        
        print("Response: ", end="", flush=True)
        async for chunk in client.inference.chat_completion(
            model_id=model_id,
            messages=messages,
            stream=True,
            max_tokens=100
        ):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # New line after streaming
        
        return True
        
    except Exception as e:
        print(f"❌ Error during responses test: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Llama Stack Integration Test")
    print("=" * 50)
    print(f"🔗 Server: http://localhost:8321")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if await test_basic_api():
        tests_passed += 1
    
    if await test_reranking_registration():
        tests_passed += 1
    
    if await test_simple_responses():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total_tests - tests_passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())