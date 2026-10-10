# experimental.ongiini.ai

A public test environment that always runs our latest Oshiwambo model, using the
same chat interface as chat.ongiini.ai. A banner on every page says it is a test
environment and links to the regular AI assistant.

```
Cloudflare tunnel  experimental.ongiini.ai → 127.0.0.1:8447
ongiini-experimental-web   nginx: page + /v1/chat, /v1/chat/clear only
ongiini-experimental       backend (webhook image), ./data-experimental, no stats loop / alerts / learn
ongiini-eval-experimental  vLLM 0.31: Gemma 4 12B + adapter "ongiini-experimental", docker bridge :8201
```

It needs ~36 GB of GPU memory, so it never runs next to training: the memory brake
may stop the model, and the training pipelines refuse to start while it runs.

## Deploy / switch the model (on the Spark)

```sh
bash deploy/experimental/serve_model.sh data/private/lora/<ADAPTER>
python3 scripts/build_experimental_chat.py --model-note "Gemma 4 12B + <ADAPTER> (<date>)"
docker compose --profile experimental up -d --build experimental experimental-web
```

Switching to a newer adapter: rerun the first two lines (the nginx serves the new
page right away; the backend needs no restart).

## Status

Live since 2026-10-10 21:19 (Namibian time) with Gemma 4 12B + T4b. The production oshiwambo
skill is replaced here (`skills/oshiwambo/SKILL.md`, via `ONGIINI_SKILLS_OVERRIDE_DIR`): the
trained model answers and translates in Oshiwambo instead of switching to English.

## One-time tunnel setup (sudo) — done 2026-10-10 (backup: config.yml.bak-2026-10-10)

Add before the catch-all rule in `/etc/cloudflared/config.yml`:

```yaml
  - hostname: experimental.ongiini.ai
    service: http://localhost:8447
```

```sh
cloudflared tunnel route dns ongiini-spark experimental.ongiini.ai
sudo systemctl restart cloudflared
```

## Pause it (before a training round)

```sh
docker compose --profile experimental stop experimental experimental-web
docker rm -f ongiini-eval-experimental
```
