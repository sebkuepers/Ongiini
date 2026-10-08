#!/usr/bin/env bash
# The model behind experimental.ongiini.ai: Gemma 4 12B + one LoRA adapter on vLLM
# 0.31 (127.0.0.1:8201), served under the name "ongiini-experimental" — the name
# the experimental backend asks for (VLLM_MODEL). Switching to a newer adapter is
# just running this again; the backend needs no restart.
#   bash deploy/experimental/serve_model.sh data/private/lora/T2_12b
# Named ongiini-eval-* on purpose: it needs ~36 GB, so it must never run next to
# training — the memory brake may stop it, and the training pipelines refuse to
# start while it runs. Stop it before a training round:
#   docker rm -f ongiini-eval-experimental
set -euo pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
[ -f "${1:?adapter dir}/adapter_config.json" ] || { echo "no adapter in $1"; exit 1; }
EVAL_NAME=ongiini-eval-experimental EVAL_PORT=8201 EVAL_RESTART=unless-stopped \
EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.30} EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-8} \
  bash deploy/eval/serve_12b_lora.sh "ongiini-experimental=$1"
