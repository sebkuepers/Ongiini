# experimental.ongiini.ai

A public test environment that always runs our latest Oshiwambo model, using the
same chat interface as chat.ongiini.ai. A banner on every page says it is a test
environment and links to the regular AI assistant.

```
Cloudflare tunnel  experimental.ongiini.ai → 127.0.0.1:8447
ongiini-experimental-web   nginx: page + /v1/chat, /v1/chat/clear only
ongiini-experimental       backend (webhook image), ./data-experimental, no stats loop / alerts / learn
ongiini-eval-experimental  vLLM 0.31: Gemma 4 12B or 31B (fp8) + adapter "ongiini-experimental", docker bridge :8201
```

It needs ~36 GB (12B) or ~58 GB (31B, fp8 at load) of GPU memory, so it never runs next to training: the memory brake
may stop the model, and the training pipelines refuse to start while it runs.

## Deploy / switch the model (on the Spark)

```sh
# all in one, waits for a free GPU, smoke-tests, reports by WhatsApp:
setsid nohup bash deploy/experimental/go_live.sh 31b data/private/lora/<ADAPTER> "Ongiini Ombuto" \
  "Gemma 4 31B + Oshiwambo training <ADAPTER> (<date>)" > data/private/experiments/go_live.log 2>&1 < /dev/null &
# or by hand:
bash deploy/experimental/serve_model.sh data/private/lora/<ADAPTER>                 # 12B adapter
EXP_MODEL=31b bash deploy/experimental/serve_model.sh data/private/lora/<ADAPTER>   # 31B adapter
python3 scripts/build_experimental_chat.py --model-name "Ongiini Ombuto" --model-note "Gemma 4 31B + <ADAPTER> (<date>)"
docker compose --profile experimental up -d --build experimental experimental-web
```

Switching to a newer adapter: rerun the first two lines (the nginx serves the new
page right away; the backend needs no restart).

## Language ratings

Under every reply visitors rate its Oshiwambo with one click (correct / small mistakes / wrong).
The backend (`ONGIINI_CHAT_FEEDBACK=1`, `/v1/chat/feedback` in `ongiini/api/chat.py`) appends
the rating with the PII-scrubbed question and answer from the server-side session to
`data-experimental/feedback/language_ratings.jsonl`; `model.txt` next to it names the live model
(written by `go_live.sh`). A changed rating is a new row; the last row per (session, turn) counts.
These are evaluation data, never training data. The start page tells visitors what is kept.

## Check the live model

The compat and tool suites can run against the live server (the base model is served next to
the adapter as gemma-4-31b / gemma-4-12b), at low concurrency so visitors are not slowed down:

```sh
BRIDGE=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)
ONGIINI_EVAL_SERVER=http://$BRIDGE:8201/v1 python3 scripts/tool_suite.py generate --model ongiini-experimental --label <LABEL> --concurrency 4
```

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
