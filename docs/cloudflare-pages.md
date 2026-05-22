# Migrate the website to Cloudflare Pages

**Why:** today `ongiini.ai` is served through the Cloudflare Tunnel from
the Spark. When the Spark loses internet (e.g. local wifi outage), the
website goes down even though it's pure static files. Moving the
website to Cloudflare Pages decouples it from the Spark entirely — the
landing page keeps working 24/7 from CF's edge, with the status
indicator in the footer honestly showing whether the AI service is
reachable.

After this change:

| Hostname / path | Where it's served from |
|---|---|
| `ongiini.ai`, `www.ongiini.ai` | Cloudflare Pages (static, always up) |
| `ongiini.ai/api/stats` | Cloudflare **Pages Function** → forwards to Spark webhook `/stats.json` |
| `api.ongiini.ai` | Cloudflare Tunnel → Spark webhook (unchanged) |

## One-time setup (you do this in the dashboard)

### Step 1 — Create the Pages project

1. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**.
2. Authorise Cloudflare to access your GitHub account, pick the
   `sebkuepers/Ongiini` repo.
3. Configure build:
   - **Production branch:** `main`
   - **Framework preset:** `None`
   - **Build command:** (leave empty)
   - **Build output directory:** `website`
   - **Root directory (advanced):** `/`
4. Click **Save and Deploy**. The first deploy takes ~30 s.

Pages gives you a preview URL like `ongiini-xxx.pages.dev` — visit it,
confirm the site renders.

### Step 2 — Add custom domain to the Pages project

1. In the Pages project → **Custom domains** → **Set up a custom domain**.
2. Enter `ongiini.ai`. Cloudflare detects it's a zone you already own
   and offers to wire it up. Confirm.
3. Repeat for `www.ongiini.ai`.

Cloudflare automatically creates the DNS records pointing the hostnames
at the Pages project (proxied, orange-cloud). You may need to **first
delete the old tunnel CNAMEs** for these hostnames (see step 3).

### Step 3 — Remove the old tunnel CNAMEs for the website

The cloudflared tunnel created CNAME records pointing `ongiini.ai` and
`www.ongiini.ai` to the tunnel UUID. Those must go before Pages can
own those hostnames.

1. Cloudflare dashboard → **ongiini.ai** zone → **DNS** → **Records**.
2. Find and delete the two CNAMEs:
   - `ongiini.ai` → `<UUID>.cfargotunnel.com`
   - `www.ongiini.ai` → `<UUID>.cfargotunnel.com`
3. **Keep** the CNAME for `api.ongiini.ai` → `<UUID>.cfargotunnel.com`
   — that's the webhook, still served from the Spark.

After this, the Pages "Add custom domain" wizard should succeed
(possibly retry it once after deletion).

### Step 4 — Clean up on the Spark

The tunnel config no longer needs to route `ongiini.ai` or
`www.ongiini.ai` (only `api.ongiini.ai`). Optional but tidy:

```sh
# On the Spark, edit /etc/cloudflared/config.yml and remove these blocks:
#   - hostname: ongiini.ai
#     service: http://localhost:18789
#   - hostname: www.ongiini.ai
#     service: http://localhost:18789
sudo systemctl restart cloudflared
```

The website container (`ongiini-website`) can also be stopped — it's
no longer reachable from the public internet:

```sh
docker compose stop website
# leave it in the compose file so we can spin it back up for local dev
```

## After the migration

- Every push to `main` triggers a Pages rebuild (~30 s). The website
  becomes a normal CD pipeline: edit `website/index.html`, push, live.
- Cloudflare Pages handles compression (Brotli + gzip), cache headers,
  image optimisation, and HTTP/3 automatically — better defaults than
  what our nginx container was doing.
- The `og-status` indicator in the footer continues to poll
  `api.ongiini.ai/status`. When the Spark is down, the dot turns red
  and the tooltip explains the pilot reality. The page itself stays up.

## Pages Function: `/api/stats`

The `/statistics` page on Pages fetches the live `/stats.json` payload
from the Spark webhook. To avoid CORS and to keep the DGX hostname out
of the browser, we use a Cloudflare Pages Function as a same-origin
proxy.

File: [`functions/api/stats.js`](../functions/api/stats.js)
Maps to: `https://ongiini.ai/api/stats`
Forwards to: `${env.STATS_API_URL}/stats.json`

### One-time setup

Set the environment variable on the Pages project so the Function knows
where to forward.

**Via the dashboard:**
1. Cloudflare → Workers & Pages → `ongiini` → **Settings** →
   **Variables and Secrets**.
2. Add to **Production** (and **Preview** if you preview-deploy):
   - Variable name: `STATS_API_URL`
   - Value: `https://api.ongiini.ai`
   - Type: Plaintext (it's not a secret — just a URL).

**Or via the Cloudflare API:**

```sh
source .env  # CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN
curl -X PATCH "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/pages/projects/ongiini" \
  -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{"deployment_configs":{"production":{"env_vars":{"STATS_API_URL":{"value":"https://api.ongiini.ai"}}},"preview":{"env_vars":{"STATS_API_URL":{"value":"https://api.ongiini.ai"}}}}}'
```

### Directory layout for Pages Functions

Cloudflare scans the `functions/` directory at the **project root**
(sibling of the assets directory you deploy):

```
Ongiini/
├── functions/
│   └── api/
│       └── stats.js     →  /api/stats
├── website/             ← deployed via `wrangler pages deploy website`
│   ├── index.html
│   └── statistics/
│       └── index.html
```

If you move `functions/` inside `website/`, wrangler won't discover it
and the route will fall through to the SPA fallback (returning the
landing page HTML — symptom: `/api/stats` returns `<!doctype html>`
instead of JSON).

### What the Function does

- Reads `env.STATS_API_URL`, returns 500 if unset.
- `fetch()`-es `${STATS_API_URL}/stats.json` with Cloudflare's edge
  caching enabled (`cf.cacheTtl: 300`).
- Sets `Cache-Control: public, max-age=300` on the response so any
  downstream cache honours the same freshness window.
- `X-Robots-Tag: noindex` on the JSON endpoint (defence-in-depth).
- Returns the upstream body unchanged with `Content-Type: application/json`.

If the upstream is unreachable or returns non-2xx, the Function returns
a small JSON error body the page can render gracefully.

## Rollback

If anything goes wrong, revert by:

1. Re-add the tunnel CNAMEs (`cloudflared tunnel route dns ongiini-spark ongiini.ai` on the Spark — but only after removing them from Pages).
2. Or delete the Pages project entirely — DNS reverts when the wizard
   does.

There's no data loss possible — the website is static files in git;
Pages is a deployment target, not a source of truth.
