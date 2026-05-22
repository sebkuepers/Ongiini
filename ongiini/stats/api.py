"""FastAPI router exposing /stats.json.

Mounted on the main app in main.py's lifespan startup. The Cloudflare
Pages Function at website/functions/api/stats.js forwards requests to
this endpoint via the existing DGX tunnel. Same-origin from the page's
perspective; no CORS dance.

The response itself is computed by `aggregator.compute()`, cached by
`cache.TTLCache` for `settings.stats_cache_ttl_seconds`. We also set
Cache-Control on the response so any HTTP intermediary (Cloudflare,
nginx) caches the same window.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import settings
from . import aggregator
from .cache import cache

router = APIRouter()


@router.get("/stats.json")
async def stats_json() -> JSONResponse:
    payload = await cache.get_or_compute(aggregator.compute)
    headers = {
        "Cache-Control": f"public, max-age={settings.stats_cache_ttl_seconds}",
    }
    return JSONResponse(content=payload, headers=headers)
