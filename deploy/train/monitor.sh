#!/usr/bin/env bash
# Log GPU load, memory and production latency every 10 s while training runs.
#   bash deploy/train/monitor.sh <container> <out.csv>
# Columns: time, gpu util %, power W, temp C, mem used GB, mem available GB,
# training container RSS, vLLM probe latency s (8-token completion).
set -u
CONTAINER="$1"; OUT="$2"
echo "time,gpu_util,power_w,temp_c,mem_used_gb,mem_avail_gb,train_rss,vllm_probe_s" > "$OUT"
while docker ps -q -f "name=^${CONTAINER}$" | grep -q .; do
  g=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  m=$(free -g | awk '/^Speicher:|^Mem:/{print $3","$7}')
  rss=$(docker stats --no-stream --format '{{.MemUsage}}' "$CONTAINER" 2>/dev/null | cut -d/ -f1 | tr -d ' ')
  t=$(curl -s -o /dev/null -w '%{time_total}' -m 60 localhost:8124/v1/chat/completions \
      -H 'content-type: application/json' \
      -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"Say ok."}],"max_tokens":8}')
  echo "$(date +%H:%M:%S),${g},${m},${rss},${t}" >> "$OUT"
  sleep 10
done
