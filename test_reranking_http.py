#!/usr/bin/env python3
"""
Test reranking API using direct HTTP calls to demonstrate functionality.
"""

import asyncio
import httpx
import json
from typing import Dict, Any


class RerankerAPITest:
    def __init__(self, base_url: str = "http://localhost:8321"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_health(self) -> bool:
        """Test server health."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/health")
            if response.status_code == 200:
                print("✅ Server is healthy!")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    async def list_models(self) -> list:
        """List all models."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            else:
                print(f"❌ Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    async def register_reranking_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a reranking model."""
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/models/reranking",
                json=model_data
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to register model: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error registering model: {e}")
            return None
    
    async def unregister_reranking_model(self, model_id: str) -> bool:
        """Unregister a reranking model."""
        try:
            response = await self.client.delete(
                f"{self.base_url}/v1/models/reranking/{model_id}"
            )
            if response.status_code == 200:
                return True
            else:
                print(f"❌ Failed to unregister model: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error unregistering model: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get model details."""
        try:
            response = await self.client.get(
                f"{self.base_url}/v1/models/{model_id}"
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"❌ Error getting model: {e}")
            return None
    
    async def run_demo(self):
        """Run the complete demonstration."""
        print("🚀 Reranking API HTTP Test")
        print("=" * 50)
        print(f"🔗 Server: {self.base_url}")
        print()
        
        # Test 1: Health check
        print("1️⃣ Testing server health...")
        if not await self.test_health():
            print("❌ Server is not running. Please start the Llama Stack server.")
            return
        
        # Test 2: List current models
        print("\n2️⃣ Listing current models...")
        models = await self.list_models()
        print(f"✅ Found {len(models)} models:")
        for model in models:
            model_type = model.get("model_type", "unknown")
            print(f"   • {model['identifier']} (type: {model_type})")
        
        # Test 3: Register a reranking model
        print("\n3️⃣ Registering a new reranking model...")
        model_data = {
            "model_id": "test-reranker",
            "provider_model_id": "rerank-v3.5",
            "provider_id": "cohere",
            "metadata": {
                "display_name": "Test Reranker Model",
                "max_tokens_per_doc": 4096,
                "description": "Demo reranking model for testing"
            }
        }
        
        registered_model = await self.register_reranking_model(model_data)
        if registered_model:
            print(f"✅ Successfully registered model: {registered_model['identifier']}")
            print(f"   Type: {registered_model['model_type']}")
            print(f"   Provider: {registered_model['provider_id']}")
        else:
            print("❌ Failed to register model")
            return
        
        # Test 4: List models again to verify
        print("\n4️⃣ Verifying model was added...")
        models = await self.list_models()
        reranking_models = [m for m in models if m.get("model_type") == "reranking"]
        print(f"✅ Found {len(reranking_models)} reranking models:")
        for model in reranking_models:
            print(f"   • {model['identifier']}")
        
        # Test 5: Get specific model details
        print("\n5️⃣ Getting model details...")
        model_details = await self.get_model("test-reranker")
        if model_details:
            print(f"✅ Model details:")
            print(f"   ID: {model_details['identifier']}")
            print(f"   Type: {model_details['model_type']}")
            print(f"   Metadata: {json.dumps(model_details['metadata'], indent=2)}")
        
        # Test 6: Unregister the model
        print("\n6️⃣ Unregistering the model...")
        if await self.unregister_reranking_model("test-reranker"):
            print("✅ Model successfully unregistered")
        
        # Test 7: Verify removal
        print("\n7️⃣ Verifying model was removed...")
        model_check = await self.get_model("test-reranker")
        if model_check is None:
            print("✅ Model properly removed")
        else:
            print("❌ Model still exists")
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("\n💡 Key takeaways:")
        print("   • New endpoints: POST /models/reranking and DELETE /models/reranking/{id}")
        print("   • Reranking models have model_type='reranking'")
        print("   • Supports Cohere, Voyage AI, and NVIDIA providers")
        print("   • Models can be registered, listed, and unregistered via API")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def test_simple_inference():
    """Test simple inference with Ollama."""
    print("\n\n💬 Testing Simple Inference...")
    print("=" * 50)
    
    client = httpx.AsyncClient(timeout=30.0)
    base_url = "http://localhost:8321"
    
    try:
        # First, register an LLM model if not already registered
        print("1️⃣ Registering LLM model...")
        model_data = {
            "model_id": "llama3.2",
            "provider_id": "ollama",
            "provider_model_id": "llama3.2",
            "model_type": "llm",
            "metadata": {}
        }
        
        response = await client.post(f"{base_url}/v1/models", json=model_data)
        if response.status_code == 200:
            print("✅ Model registered successfully")
        else:
            print(f"⚠️ Model registration returned: {response.status_code}")
        
        # Test chat completion
        print("\n2️⃣ Testing chat completion...")
        chat_data = {
            "model_id": "llama3.2",
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer in one word only."}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = await client.post(
            f"{base_url}/v1/inference/chat-completion",
            json=chat_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat completion succeeded!")
            print(f"   Raw response: {json.dumps(result, indent=2)}")
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                print(f"   Response content: {content}")
            elif "completion_message" in result:
                content = result["completion_message"]["content"]
                print(f"   Response content: {content}")
        else:
            print(f"❌ Chat completion failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ Error during inference test: {e}")
    
    finally:
        await client.aclose()


async def main():
    """Run all tests."""
    test = RerankerAPITest()
    try:
        await test.run_demo()
        await test_simple_inference()
    finally:
        await test.close()


if __name__ == "__main__":
    asyncio.run(main())