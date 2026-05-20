FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the CPU-only torch wheel BEFORE the rest of requirements.txt so
# that sentence-transformers (transitively pulled by mem0's huggingface
# embedder) doesn't accidentally pull the CUDA torch build. We want all
# of the Spark's GPU memory dedicated to vLLM serving Gemma 4 — the
# 22M-param embedding model fits comfortably on CPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the sentence-transformers model into the image so the
# first container start doesn't have to hit huggingface.co. Keeps
# cold-start latency predictable and works even when the Spark loses
# its uplink (the wifi-watchdog scenario).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app

# Run as a non-root user matching the host's primary user (UID 1000), so
# files written into the bind-mounted /data volume are owned by `nexus`
# rather than root.
RUN groupadd --system --gid 1000 ongiini \
 && useradd  --system --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin ongiini \
 && chown -R ongiini:ongiini /app

# Sentence-transformers caches into the user's home dir by default. Make
# sure that path exists and is writable for the non-root user, and reuse
# the model we pre-downloaded during the build.
ENV HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache/sentence_transformers
RUN mkdir -p /app/.hf_cache \
 && cp -r /root/.cache/huggingface/* /app/.hf_cache/ 2>/dev/null || true \
 && chown -R ongiini:ongiini /app/.hf_cache

USER ongiini

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
