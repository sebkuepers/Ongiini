#!/usr/bin/env bash
# Temporary vLLM server for evaluation: Gemma 4 12B + LoRA adapters, served the
# way production serves Gemma 4 26B (same image, chat template, tool-call and
# reasoning parsers, image limits), on 127.0.0.1:8200. The base model is
# "gemma-4-12b"; each adapter is selectable by its name (vLLM multi-LoRA).
# Container name ongiini-eval-* so the memory brake and production guard cover it.
#   bash deploy/eval/serve_12b_lora.sh t2=data/private/lora/T2_12b [name=dir ...]
#   docker rm -f ongiini-eval-vllm12b
set -euo pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
IMAGE=vllm/vllm-openai:gemma4-0505-arm64-cu130
MODEL_DIR="$HOME/models/gemma-4-12b-it-bf16"
TEMPLATE=deploy/spark/tool_chat_template_gemma4.jinja
mods=() mounts=()
for spec in "$@"; do
  name=${spec%%=*} dir=${spec#*=}
  mounts+=(-v "$PWD/$dir:/adapters/$name:ro")
  mods+=("$name=/adapters/$name")
done
docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1 || true
docker run -d --name ongiini-eval-vllm12b --gpus all --ipc host --shm-size 16g \
  -p 127.0.0.1:8200:8000 \
  -v "$MODEL_DIR:/models/gemma-4-12b:ro" -v "$PWD/$TEMPLATE:/templates/chat.jinja:ro" "${mounts[@]}" \
  "$IMAGE" \
  --model /models/gemma-4-12b --served-model-name gemma-4-12b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${EVAL_GPU_MEM_UTIL:-0.30}" \
  --reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4 \
  --chat-template /templates/chat.jinja \
  --limit-mm-per-prompt '{"image": 4, "audio": 0}' --mm-processor-kwargs '{"max_soft_tokens": 280}' \
  --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules "${mods[@]}"
echo "waiting for :8200"
for _ in $(seq 1 90); do
  curl -sf -m 5 localhost:8200/v1/models >/dev/null && { echo "ready"; curl -s localhost:8200/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"; exit 0; }
  docker ps -q -f name=^ongiini-eval-vllm12b$ | grep -q . || { echo "container exited"; docker logs --tail 30 ongiini-eval-vllm12b 2>&1; exit 1; }
  sleep 10
done
echo "not ready after 15 min"; exit 1
