# LoRA training image for the DGX Spark (paper 3 experiments).
# Builds on the production vLLM image because it already ships a PyTorch
# built for the GB10 (sm_121, CUDA 13, aarch64), transformers, accelerate and
# bitsandbytes; adds the fine-tuning libraries only.
FROM vllm/vllm-openai:gemma4-0505-arm64-cu130
# transformers is upgraded: Gemma 4 12B uses the newer `gemma4_unified`
# architecture, unknown to the 5.8 release shipped with the vLLM image.
RUN pip install --no-cache-dir -U "transformers" && \
    pip install --no-cache-dir "peft>=0.17" "trl>=0.24" "datasets>=3.0"
WORKDIR /work
ENTRYPOINT []
