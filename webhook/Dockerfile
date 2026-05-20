FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Run as a non-root user matching the host's primary user (UID 1000), so
# files written into the bind-mounted /data volume are owned by `nexus`
# rather than root.
RUN groupadd --system --gid 1000 ongiini \
 && useradd  --system --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin ongiini \
 && chown -R ongiini:ongiini /app

USER ongiini

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
