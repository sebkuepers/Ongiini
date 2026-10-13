#!/usr/bin/env bash
# The model behind experimental.ongiini.ai: Gemma 4 (12B or 31B) + one LoRA adapter on
# vLLM 0.31 (docker bridge :8201), served under the name "ongiini-experimental" — the name
# the experimental backend asks for (VLLM_MODEL). Switching to a newer adapter is just
# running this again; the backend needs no restart.
#   bash deploy/experimental/serve_model.sh data/private/lora/T4b_12b          # 12B
#   EXP_MODEL=31b bash deploy/experimental/serve_model.sh data/private/lora/T4b_31b
# The 31B (62.5 GB in bf16) is quantised to fp8 at load and gets 0.48 of the memory (~58 GB).
# Measured 2026-10-13: weights 33.6 GiB; one 32k-token request needs 15.8 GiB KV cache; 0.40
# left only 13.9 GiB and vLLM refused to start -> 0.48 (~23 GiB KV). The 12B needs ~36 GB. The base model stays selectable
# as gemma-4-12b / gemma-4-31b, so the compat and tool suites can run against this server
# (ONGIINI_EVAL_SERVER=http://<bridge>:8201/v1).
# Named ongiini-eval-* on purpose: it must never run next to training — the memory brake
# may stop it, and the training pipelines refuse to start while it runs. Stop it before a
# training round:
#   docker rm -f ongiini-eval-experimental
set -euo pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
[ -f "${1:?adapter dir}/adapter_config.json" ] || { echo "no adapter in $1"; exit 1; }
# bound to the docker bridge (docker0): the experimental backend reaches it via host.docker.internal,
# the LAN does not (127.0.0.1 is unreachable from containers)
BRIDGE=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)
[ -n "$BRIDGE" ] || { echo "no docker0 address"; exit 1; }
case "${EXP_MODEL:-12b}" in
  12b) export EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.30} ;;
  31b) export EVAL_MODEL_DIR="$HOME/models/gemma-4-31b-it-bf16" EVAL_SERVED_NAME=gemma-4-31b EVAL_QUANT=fp8 \
              EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.48} ;;
  *) echo "EXP_MODEL must be 12b or 31b"; exit 1 ;;
esac
EVAL_NAME=ongiini-eval-experimental EVAL_PORT=8201 EVAL_RESTART=unless-stopped EVAL_BIND=$BRIDGE \
EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-8} \
  bash deploy/eval/serve_12b_lora.sh "ongiini-experimental=$1"
