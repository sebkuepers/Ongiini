# Meyabase translation service (see serve.py). Build context: deploy/meyabase.
# CPU-only: the Marian models are ~75M parameters, so the Spark's GPU stays
# with vLLM. Meyabase's inference code is pinned to a reviewed commit; model
# weights are downloaded at runtime into the /cache volume.
FROM python:3.12-slim AS src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG MEYABASE_REPO=https://github.com/axelmukwena/meyabase-translate-inference
ARG MEYABASE_COMMIT=f559eb8
RUN git clone "$MEYABASE_REPO" /src && cd /src && git checkout "$MEYABASE_COMMIT"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/cache HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=2
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 \
 && pip install --no-cache-dir transformers==4.57.6 sentencepiece==0.2.1 sacremoses==0.1.1 \
      huggingface-hub==0.36.2 fastapi==0.115.14 "uvicorn[standard]==0.34.3"
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=src /src/translation ./translation
COPY serve.py ./
USER app
EXPOSE 8080
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-server-header"]
