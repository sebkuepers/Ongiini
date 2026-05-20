# Cloudflare hardening — one-time setup

Click-through checklist for the Cloudflare dashboard at
[dash.cloudflare.com](https://dash.cloudflare.com). Free plan covers
all of this.

These rules sit in front of the Cloudflare Tunnel and apply to traffic
hitting `ongiini.ai`, `www.ongiini.ai`, and `api.ongiini.ai`.

## 1. Bot Fight Mode (recommended)

**Why:** automatically challenges low-effort automated traffic before
it reaches our origin. Catches the long-tail of script-kiddie scans.

- **Cloudflare dashboard → ongiini.ai → Security → Bots**
- Turn on **Bot Fight Mode** (free).

## 2. Rate-limit `api.ongiini.ai/*` (recommended)

**Why:** caps webhook flood / token-exhaustion attacks at the edge,
before they even reach the FastAPI process. Complements the per-MSISDN
limiter inside the webhook.

- **Cloudflare dashboard → ongiini.ai → Security → WAF → Rate limiting rules**
- Click **Create rule**:
  - **Rule name:** `webhook-burst-limit`
  - **If incoming requests match:** Custom filter expression
    - Field: `Hostname` — Operator: `equals` — Value: `api.ongiini.ai`
  - **Then take action:** `Block`
  - **For duration:** `10 minutes`
  - **When requests per:** `1 minute` exceed `120`
  - **Counting characteristics:** `IP source address`
- Save & deploy.

120 req/min/IP is generous (Meta will be the main caller and won't
come close). Adjust down if you want stricter.

## 3. Restrict `api.ongiini.ai` to Meta IPs (optional, stricter)

**Why:** belt-and-braces on top of the App Secret signature check.
Only Meta's WhatsApp Cloud API IPs can POST to the webhook.

Meta publishes their address ranges at
[developers.facebook.com/docs/whatsapp/api/security](https://developers.facebook.com/docs/whatsapp/api/security/).
The IP list rotates occasionally; revisit every few months.

- **Cloudflare dashboard → ongiini.ai → Security → WAF → Custom rules**
- Click **Create rule**:
  - **Rule name:** `webhook-only-meta-ips`
  - **If incoming requests match:** Custom expression
    ```
    (http.host eq "api.ongiini.ai") and
    not (ip.src in {31.13.24.0/21 31.13.64.0/18 66.220.144.0/20
                    69.63.176.0/20 69.171.224.0/19 74.119.76.0/22
                    103.4.96.0/22 129.134.0.0/16 157.240.0.0/16
                    173.252.64.0/18 179.60.192.0/22 185.60.216.0/22
                    204.15.20.0/22 2401:db00::/32 2620:0:1c00::/40
                    2803:6080::/32 2a03:2880::/32 2a03:2880:f000::/48
                    2a03:2880:f100::/48 2a03:2880:f200::/48
                    2620:10d:c082::/47 2620:10d:c081::/48})
    ```
  - **Action:** `Block`
- Save & deploy.

**Caveat:** if you hit "this looks fine, deploy" and Meta has updated
their ranges since this file was written, the verification probe from
the Meta dashboard will fail. To debug: temporarily disable this rule,
re-verify, then re-enable with updated ranges.

## 4. Sensible "Security Level"

- **Cloudflare dashboard → ongiini.ai → Security → Settings**
- Set **Security Level** to **Medium** (default). High is over-eager
  on the marketing page and challenges normal users.

## 5. Optional but cheap wins

- **Email Address Obfuscation: On** — scrambles `mailto:` links on the
  website so they're harder for scrapers to harvest.
- **Browser Integrity Check: On** — blocks browsers with obviously
  fake user agents.
- **Always Use HTTPS: On** — automatic 301 from http to https.

Find these under **Security → Settings** and **SSL/TLS → Edge
Certificates**.

## What's intentionally NOT here

- **Origin authentication** (mutual TLS between CF and origin) — our
  origin is a Cloudflare Tunnel, so the Tunnel binary already proves
  identity to Cloudflare. Mutual TLS would be redundant.
- **WAF Managed Rulesets** — paid tier only.
- **DDoS Alerts** — paid tier (Pro and up).
