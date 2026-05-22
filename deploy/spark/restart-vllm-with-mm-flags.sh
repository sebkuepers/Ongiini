#!/usr/bin/env bash
# Restart gemma4-vllm with the multimodal-recommended flags.
#
# What this changes vs the v1 command in the README:
#   + --chat-template <gemma4 tool chat template>
#       Fixes vLLM #41452 ("Gemma4 can't process images in tool message")
#       — lets us pass tools= alongside image_url content again.
#   + --limit-mm-per-prompt '{"image": 4, "audio": 0}'
#       Required to enable image profiling and explicitly disables the
#       audio tower allocation we don't use.
#   + --mm-processor-kwargs '{"max_soft_tokens": 280}'
#       Explicit vision token budget (default is 280, made explicit for
#       reproducibility).
#   + --hf-overrides '{"vision_config":{"torch_dtype":"bfloat16"}}'
#       Forces the vision tower to bf16 even when the rest of the
#       engine runs NVFP4. Works around vLLM #40290 where the loader
#       silently re-casts vision_tower to fp16 and overflows.
#   + --max-num-seqs 16
#       Tames concurrent multimodal allocations on the Spark.
#   + --async-scheduling
#       Recommended by the vLLM Gemma 4 recipe for MoE multimodal.
#   + --enable-prompt-tokens-details
#       Populates `prompt_tokens_details.cached_tokens` in the
#       OpenAI-compatible usage response so the webhook can bill
#       users only for uncached prompt + completion. With prefix
#       caching ON by default, the static SYSTEM_PROMPT + TOOLS +
#       product.md hits cache after first request and is reported
#       as cached. Without this flag, vLLM reports prompt_tokens
#       in full and the user gets over-billed for every turn's
#       static overhead.
#
# Service interruption: ~3-4 minutes during cold model load.

set -euo pipefail

CONTAINER=gemma4-vllm
IMAGE=vllm/vllm-openai:gemma4-0505-arm64-cu130
MODEL_DIR=/home/nexus/models/gemma-4-26b-a4b-nvfp4
TEMPLATE_PATH=/vllm-workspace/examples/tool_chat_template_gemma4.jinja

echo "==> stopping $CONTAINER"
docker stop "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true

echo "==> starting $CONTAINER with multimodal flags"
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --gpus all --ipc host --shm-size 64gb \
  -p 8124:8000 \
  -v "$MODEL_DIR:/models/gemma4" \
  "$IMAGE" \
  --model /models/gemma4 \
  --served-model-name gemma-4-26b \
  --host 0.0.0.0 --port 8000 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.70 \
  --moe-backend marlin \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --chat-template "$TEMPLATE_PATH" \
  --limit-mm-per-prompt '{"image": 4, "audio": 0}' \
  --mm-processor-kwargs '{"max_soft_tokens": 280}' \
  --hf-overrides '{"vision_config":{"torch_dtype":"bfloat16"}}' \
  --async-scheduling \
  --enable-prompt-tokens-details

echo "==> waiting for /v1/models to respond (up to 5 minutes)"
for i in $(seq 1 60); do
  if curl -sf --max-time 3 http://127.0.0.1:8124/v1/models >/dev/null 2>&1; then
    echo "==> ready after ${i}x5s = $((i*5))s"
    curl -s http://127.0.0.1:8124/v1/models | head -c 200
    echo
    exit 0
  fi
  sleep 5
done

echo "==> ERROR: /v1/models never came back" >&2
docker logs --tail 30 "$CONTAINER"
exit 1
