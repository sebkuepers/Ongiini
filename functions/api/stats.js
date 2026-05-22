// Cloudflare Pages Function: /api/stats
//
// Proxies the browser's request to the FastAPI webhook on the DGX,
// without exposing the tunnel hostname client-side and without
// triggering CORS. The DGX hostname is read from the env var
// STATS_API_URL configured in Cloudflare Pages settings (must include
// scheme + host; the function appends "/stats.json").
//
// Caches the upstream response at the CDN edge for 5 minutes regardless
// of who calls — pages always serve a recent snapshot even if a burst
// of visitors hits at once.

const UPSTREAM_PATH = "/stats.json";
const CDN_CACHE_SECONDS = 300;

export async function onRequestGet({ env }) {
  const base = env.STATS_API_URL;
  if (!base) {
    return jsonError(500, "STATS_API_URL is not configured on this deployment.");
  }
  const upstream = base.replace(/\/+$/, "") + UPSTREAM_PATH;

  let res;
  try {
    res = await fetch(upstream, {
      method: "GET",
      cf: {
        cacheTtl: CDN_CACHE_SECONDS,
        cacheEverything: true,
      },
    });
  } catch (err) {
    return jsonError(502, "Upstream unreachable.", String(err));
  }

  if (!res.ok) {
    return jsonError(res.status, "Upstream returned a non-OK response.");
  }

  // Pass through the body as-is, replace Cache-Control so the browser
  // and any downstream cache honour our chosen freshness window
  // regardless of what the FastAPI endpoint sets.
  const body = await res.text();
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": `public, max-age=${CDN_CACHE_SECONDS}`,
      // Belt-and-braces noindex for the JSON endpoint itself.
      "X-Robots-Tag": "noindex",
    },
  });
}

function jsonError(status, message, detail) {
  const body = detail
    ? JSON.stringify({ error: message, detail })
    : JSON.stringify({ error: message });
  return new Response(body, {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}
