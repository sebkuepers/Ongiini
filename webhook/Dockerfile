FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg decodes WhatsApp's OGG/Opus voice notes (and any other audio
# format the user might send). System package is the simplest path —
# ffmpeg-python wrappers all shell out to the ffmpeg binary anyway.
#
# espeak-ng is dev-only — used by webhook/tests/audio_smoke.py to
# synthesise short EN/AF speech samples so we can verify the Whisper
# pipeline transcribes real speech (not just survives empty audio).
# Tiny package (~3 MB) and unused at runtime so the cost is negligible.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
 && rm -rf /var/lib/apt/lists/*

# Install the CPU-only torch wheel BEFORE the rest of requirements.txt so
# that sentence-transformers (transitively pulled by mem0's huggingface
# embedder) doesn't accidentally pull the CUDA torch build. We want all
# of the Spark's GPU memory dedicated to vLLM serving Gemma 4 — the
# 22M-param embedding model fits comfortably on CPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

COPY webhook/requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the sentence-transformers model into the image so the
# first container start doesn't have to hit huggingface.co. Keeps
# cold-start latency predictable and works even when the Spark loses
# its uplink (the wifi-watchdog scenario).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY webhook/app ./app
# Owela framework lives at the repo root; copy it next to ``app`` so
# the application's ``from owela import ...`` imports resolve at runtime.
COPY owela ./owela

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
