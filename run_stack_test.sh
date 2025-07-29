#!/bin/bash
set -e

echo "🚀 Starting Llama Stack build and test process..."

# Step 1: Check if Ollama is running
echo "1️⃣ Checking Ollama status..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first."
    exit 1
fi

if ! ollama list &> /dev/null; then
    echo "❌ Ollama is not running. Please start Ollama first."
    exit 1
fi

echo "✅ Ollama is running"

# Step 2: Check for llama3.2 model
echo "2️⃣ Checking for llama3.2 model..."
if ! ollama list | grep -q "llama3.2"; then
    echo "📥 Downloading llama3.2 model..."
    ollama pull llama3.2
else
    echo "✅ llama3.2 model is available"
fi

# Step 3: Try to build a minimal stack
echo "3️⃣ Building Llama Stack with minimal configuration..."

# Create a minimal run.yaml for testing
cat > /tmp/test_run.yaml << 'EOF'
built_at: '2025-07-29T04:14:22.605000'
image_type: venv
distribution_spec:
  description: Starter pack including all default core dependencies
  providers:
    inference:
    - provider_id: ollama
      config:
        url: http://localhost:11434
      provider_type: remote::ollama
    models:
    - provider_id: ollama
      config:
        url: http://localhost:11434
      provider_type: remote::ollama
    memory:
    - provider_id: meta-reference-memory
      config: {}
      provider_type: inline::meta-reference
    safety:
    - provider_id: meta-reference-safety
      config: {}
      provider_type: inline::meta-reference
    agents:
    - provider_id: meta-reference-agents
      config: {}
      provider_type: inline::meta-reference
    telemetry:
    - provider_id: meta-reference-telemetry
      config: {}
      provider_type: inline::meta-reference
    reranker:
    - provider_id: cohere
      config:
        api_key: ${env.COHERE_API_KEY:}
      provider_type: remote::cohere
    - provider_id: voyage
      config:
        api_key: ${env.VOYAGE_API_KEY:}
      provider_type: remote::voyage
    - provider_id: nvidia
      config:
        api_key: ${env.NVIDIA_API_KEY:}
      provider_type: remote::nvidia
  routing_table:
    inference:
      routing_map: {}
    memory:
      routing_map: {}
    safety:
      routing_map: {}
    agents:
      routing_map: {}
    telemetry:
      routing_map: {}
    reranker:
      routing_map: {}
    models:
      routing_map: {}
  model_parallel_dimensions: {}
apis:
- inference
- memory
- safety
- agents
- telemetry
- reranker
- models
metadata:
  conda_env: null
  run_config_path: /tmp/test_run.yaml
  image_name: /tmp/test_stack
  venv_path: /tmp/test_stack
EOF

echo "4️⃣ Starting Llama Stack server..."

# Run the stack in the background
LLAMA_STACK_PORT=5001 /Users/raghu/.local/bin/uv run --python 3.12 python -m llama_stack.distribution.server.server --yaml-config /tmp/test_run.yaml &
STACK_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 10

# Check if server is running
if ! curl -s http://localhost:5001/health > /dev/null; then
    echo "❌ Server failed to start"
    kill $STACK_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Llama Stack server is running"

# Step 5: Run integration tests
echo "5️⃣ Running integration tests..."
cd /Users/raghu/dev/llama-stack
/Users/raghu/.local/bin/uv run --python 3.12 python test_reranking_integration.py

# Cleanup
echo "🧹 Cleaning up..."
kill $STACK_PID 2>/dev/null || true
rm -f /tmp/test_run.yaml

echo "✅ Integration test completed!"
EOF