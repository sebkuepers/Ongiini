#!/usr/bin/env bash
# Temporary vLLM server for evaluation: Gemma 4 12B + LoRA adapters, served the
# way production serves Gemma 4 26B (same image, chat template, tool-call and
# reasoning parsers, image limits), on 127.0.0.1:8200. The base model is
# "gemma-4-12b"; each adapter is selectable by its name (vLLM multi-LoRA).
# Container name ongiini-eval-* so the memory brake and production guard cover it.
#   bash deploy/eval/serve_12b_lora.sh t2=data/private/lora/T2_12b [name=dir ...]
#   docker rm -f ongiini-eval-vllm12b
# EVAL_NAME / EVAL_PORT / EVAL_RESTART reuse it for experimental.ongiini.ai
# (deploy/experimental/serve_model.sh).
set -euo pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
# The production image (vLLM 0.20.2) cannot serve the dense 12B (gemma4_unified);
# vLLM 0.31.0 supports Gemma4UnifiedForConditionalGeneration natively (2026-10-08).
IMAGE=${EVAL_VLLM_IMAGE:-vllm/vllm-openai:v0.31.0-aarch64}
MODEL_DIR="$HOME/models/gemma-4-12b-it-bf16"
TEMPLATE=deploy/spark/tool_chat_template_gemma4.jinja
mods=() mounts=()
for spec in "$@"; do
  name=${spec%%=*} dir=${spec#*=}
  mounts+=(-v "$PWD/$dir:/adapters/$name:ro")
  mods+=("$name=/adapters/$name")
done
NAME=${EVAL_NAME:-ongiini-eval-vllm12b} PORT=${EVAL_PORT:-8200}
BIND=${EVAL_BIND:-127.0.0.1}  # experimental.ongiini.ai: the docker bridge, so containers (not the LAN) reach it
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart "${EVAL_RESTART:-no}" --gpus all --ipc host --shm-size 16g \
  -p "$BIND:$PORT:8000" \
  -v "$MODEL_DIR:/models/gemma-4-12b:ro" -v "$PWD/$TEMPLATE:/templates/chat.jinja:ro" "${mounts[@]}" \
  --entrypoint vllm "$IMAGE" serve /models/gemma-4-12b \
  --model-impl "${EVAL_MODEL_IMPL:-auto}" \
  --served-model-name gemma-4-12b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs "${EVAL_MAX_SEQS:-16}" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${EVAL_GPU_MEM_UTIL:-0.30}" \
  --reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4 \
  --chat-template /templates/chat.jinja \
  --limit-mm-per-prompt '{"image": 4, "audio": 0}' --mm-processor-kwargs '{"max_soft_tokens": 280}' \
  --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules "${mods[@]}"
echo "waiting for :$PORT"
for _ in $(seq 1 90); do
  curl -sf -m 5 "$BIND:$PORT/v1/models" >/dev/null && { echo "ready"; curl -s "$BIND:$PORT/v1/models" | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"; exit 0; }
  docker ps -q -f name=^$NAME$ | grep -q . || { echo "container exited"; docker logs --tail 30 "$NAME" 2>&1; exit 1; }
  sleep 10
done
echo "not ready after 15 min"; exit 1
