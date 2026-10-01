#!/usr/bin/env python3
"""One source of truth for the search / sharing metadata of ongiini.ai.

For every public page this script owns a marked block in <head>:

    <!-- seo:start --> … title, description, robots, canonical, Open Graph,
                         Twitter, icons, theme-color, JSON-LD … <!-- seo:end -->

and it writes website/sitemap.xml. Edit the PAGES table below, not the
HTML heads. The home page FAQ JSON-LD is generated from the visible FAQ
data in website/index.html (``var FAQ = [...]``), so the markup can never
drift from what people read.

    python3 scripts/build_site_seo.py            # rewrite heads + sitemap
    python3 scripts/build_site_seo.py --check    # CI / pre-commit: fail on drift or rule breaks

The research page is generated from scripts/research_page_template.html;
its block is written into the template too, so a rebuild keeps it.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "website"
SITE = "https://ongiini.ai"
BRAND = "Ongiini AI"
ICON_512 = f"{SITE}/assets/ongiini-icon-512-v2.png"
GITHUB = "https://github.com/sebkuepers/Ongiini"
ARXIV = "https://arxiv.org/abs/2609.31727"

# path → page. "file" is relative to website/. og = image in assets/og/.
PAGES: list[dict] = [
    {"url": "/", "file": "index.html", "priority": "1.0", "og": "home",
     "title": "Ongiini AI — Free AI assistant on WhatsApp for Namibia",
     "description": "Ongiini AI is a free AI assistant on WhatsApp for Namibia — school, jobs, health, "
                    "farming and daily life, in English and Afrikaans. Ask in text or send a photo.",
     "og_alt": "Ongiini AI — help for life's questions, on WhatsApp", "locale_alt": "af_NA",
     "robots": "index, follow, max-snippet:-1, max-image-preview:large", "home": True},
    {"url": "/contribute/", "file": "contribute/index.html", "priority": "0.9", "og": "contribute",
     "title": "Help with Oshiwambo translations — Ongiini AI",
     "description": "Speak Oshindonga or Oshikwanyama? Check translations in two minutes or translate "
                    "sentences on WhatsApp, for a free and open Oshiwambo dataset.",
     "og_alt": "Help Ongiini AI speak Oshiwambo", "locale_alt": "af_NA", "crumb": "Contribute"},
    {"url": "/research/", "file": "research/index.html", "priority": "0.8", "og": "research",
     "template": "scripts/research_page_template.html",
     "title": "Ongiini-Eval-OW: Oshiwambo MT benchmark — Ongiini AI",
     "description": "An open English → Oshindonga and Oshikwanyama translation benchmark: 600 sentences, "
                    "native-speaker references, 11 phenomena. Get your model scored.",
     "og_alt": "How well do machines translate into Oshiwambo?", "crumb": "Research", "research": True},
    {"url": "/statistics/", "file": "statistics/index.html", "priority": "0.7", "og": "statistics",
     "title": "Live usage statistics — Ongiini AI",
     "description": "Live, aggregate statistics on how people in Namibia use Ongiini AI, the free WhatsApp "
                    "AI assistant: users, topics and languages. Never individual chats.",
     "og_alt": "How Namibia uses Ongiini AI", "crumb": "Statistics"},
    {"url": "/rate/", "file": "rate/index.html", "priority": "0.7", "og": "rate",
     "title": "Check Oshiwambo translations — Ongiini AI",
     "description": "Native speaker of Oshindonga or Oshikwanyama? Check whether translations say the same "
                    "as the English: five sentences, about two minutes, started on WhatsApp.",
     "og_alt": "Does it say the same? Check Oshiwambo translations", "crumb": "Check translations"},
    {"url": "/slides/", "file": "slides/index.html", "priority": "0.5", "og": "slides",
     "title": "Slides for volunteers — Ongiini AI",
     "description": "A short, volunteer-friendly slide deck that explains Ongiini AI, the free WhatsApp AI "
                    "assistant for Namibia. Present it on screen or print it to PDF.",
     "og_alt": "Ongiini AI, explained", "locale_alt": "af_NA", "crumb": "Slides"},
    {"url": "/slides/guide/", "file": "slides/guide/index.html", "priority": "0.4", "og": "slides",
     "title": "Volunteer presenter guide — Ongiini AI",
     "description": "How to present the Ongiini AI slides to your community: what each slide is about, "
                    "what to say, and the questions people usually ask afterwards.",
     "og_alt": "Ongiini AI, explained", "locale_alt": "af_NA",
     "crumbs": [("Slides", "/slides/"), ("Presenter guide", "/slides/guide/")]},
    {"url": "/privacy/", "file": "privacy/index.html", "priority": "0.2", "og": "home",
     "title": "Privacy policy — Ongiini AI",
     "description": "How Ongiini AI handles your data: what we process from WhatsApp and the web chat, why, "
                    "for how long, who else sees it, and your rights under the GDPR.",
     "og_alt": "Ongiini AI", "crumb": "Privacy policy"},
    {"url": "/terms/", "file": "terms/index.html", "priority": "0.2", "og": "home",
     "title": "Terms of service — Ongiini AI",
     "description": "The terms for using Ongiini AI, the free AI assistant on WhatsApp for Namibia: what "
                    "the service is, acceptable use, and the limits of AI answers.",
     "og_alt": "Ongiini AI", "crumb": "Terms of service"},
    {"url": "/imprint/", "file": "imprint/index.html", "priority": "0.2", "og": "home",
     "title": "Imprint — Ongiini AI",
     "description": "Legal notice for ongiini.ai: who operates Ongiini AI, the free AI assistant on WhatsApp "
                    "for Namibia, who is responsible for the content, and how to reach us.",
     "og_alt": "Ongiini AI", "crumb": "Imprint"},
    # separate Cloudflare Pages projects (chat.ongiini.ai, learn.ongiini.ai)
    {"url": "https://chat.ongiini.ai/", "file": "chat-app/index.html", "priority": "0.6", "og": "home",
     "title": "Chat in your browser — Ongiini AI",
     "description": "Try Ongiini AI, the free AI assistant for Namibia, right in your browser: no app, no "
                    "login, no signup. Ask anything in English or Afrikaans.",
     "og_alt": "Ongiini AI", "locale_alt": "af_NA"},
    {"url": "https://learn.ongiini.ai/", "file": "learn-app/index.html", "priority": "0.5", "og": "home",
     "title": "Learn with AI, starting with Afrikaans — Ongiini AI",
     "description": "Ongiini Learn: a personalised AI learning experience from Ongiini AI, starting with "
                    "Afrikaans. No app, no signup. Learn at your own pace on your phone.",
     "og_alt": "Ongiini AI"},
]

# Tags inside <head> that the block replaces (removed wherever they sit outside it).
OWNED = [
    r'<title>.*?</title>',
    r'<meta\s+name="(?:description|robots|theme-color|twitter:[a-z:]+)"[^>]*>',
    r'<meta\s+property="og:[a-z:_]+"[^>]*>',
    r'<link\s+rel="(?:canonical|icon|apple-touch-icon|shortcut icon)"[^>]*>',
    r'<script type="application/ld\+json">.*?</script>',
]
BLOCK_RE = re.compile(r"[ \t]*<!-- seo:start -->.*?<!-- seo:end -->\n?", re.S)


def absolute(url: str) -> str:
    return url if url.startswith("http") else SITE + url


def esc(s: str) -> str:
    return html.escape(s, quote=True)


# ── JSON-LD ──────────────────────────────────────────────────────────

def faq_from_home(index_html: str) -> list[tuple[str, str]]:
    """The visible FAQ (English) from ``var FAQ = [ { en: ["Q", "A"], … } ]``."""
    m = re.search(r"var FAQ = \[(.*?)\n\s*\];", index_html, re.S)
    if not m:
        raise SystemExit("FAQ data not found in website/index.html")
    s = r'"(?:[^"\\]|\\.)*"'
    pairs = re.findall(rf"en:\s*\[\s*({s})\s*,\s*({s})\s*\]", m.group(1))
    return [(json.loads(q), json.loads(a)) for q, a in pairs]


def breadcrumbs(page: dict) -> dict | None:
    trail = page.get("crumbs") or ([(page["crumb"], page["url"])] if page.get("crumb") else None)
    if not trail:
        return None
    items = [("Ongiini AI", "/")] + trail
    return {"@context": "https://schema.org", "@type": "BreadcrumbList",
            "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n, "item": absolute(u)}
                                for i, (n, u) in enumerate(items)]}


def jsonld(page: dict) -> list[dict]:
    url = absolute(page["url"])
    out: list[dict] = []
    if page.get("home"):
        out.append({"@context": "https://schema.org", "@type": "Organization",
                    "name": "Common Intelligence Foundation", "alternateName": BRAND, "url": SITE + "/",
                    "logo": ICON_512,
                    "description": "A non-profit foundation, being established in Estonia, that builds and "
                                   "operates free AI access for communities locked out of AI. Ongiini AI is "
                                   "its first project.",
                    "foundingLocation": {"@type": "Country", "name": "Estonia"},
                    "sameAs": ["https://common-intelligence.org", GITHUB]})
        out.append({"@context": "https://schema.org", "@type": "WebSite", "name": BRAND,
                    "alternateName": "Ongiini", "url": SITE + "/", "inLanguage": ["en", "af"]})
        out.append({"@context": "https://schema.org", "@type": "WebApplication", "name": BRAND,
                    "url": SITE + "/", "applicationCategory": "CommunicationApplication",
                    "operatingSystem": "WhatsApp", "isAccessibleForFree": True,
                    "description": page["description"], "inLanguage": ["en", "af"],
                    "offers": {"@type": "Offer", "price": "0", "priceCurrency": "NAD"},
                    "audience": {"@type": "Audience", "geographicArea": {"@type": "Country", "name": "Namibia"}},
                    "creator": {"@type": "Organization", "name": "Common Intelligence Foundation"}})
        faq = faq_from_home((WEB / "index.html").read_text())
        out.append({"@context": "https://schema.org", "@type": "FAQPage",
                    "mainEntity": [{"@type": "Question", "name": q,
                                    "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in faq]})
        return out
    if page.get("research"):
        people = [{"@type": "Person", "name": n} for n in ("Sebastian Küpers", "Kaarina Shoozi",
                                                            "Elizabeth Hamukwaya")]
        out.append({"@context": "https://schema.org", "@type": "Dataset", "name": "Ongiini-Eval-OW",
                    "alternateName": "Ongiini Oshiwambo evaluation set",
                    "description": "An English → Oshindonga and English → Oshikwanyama machine-translation "
                                   "benchmark with native-speaker references: 600 English source sentences in "
                                   "the register Namibians write, tagged for 11 linguistic phenomena, with a "
                                   "180-sentence blind split. The English sources are public; references are "
                                   "released with v1.0.",
                    "url": url, "sameAs": f"{GITHUB}/tree/main/data/oshiwambo_eval",
                    "keywords": ["machine translation", "benchmark", "Oshiwambo", "Oshindonga",
                                 "Oshikwanyama", "Namibia", "low-resource languages"],
                    "inLanguage": ["en", "ng", "kj"], "isAccessibleForFree": True,
                    "license": "https://creativecommons.org/licenses/by/4.0/",
                    "creator": people, "citation": ARXIV,
                    "spatialCoverage": {"@type": "Place", "name": "Namibia"},
                    "distribution": [{"@type": "DataDownload", "encodingFormat": "text/tab-separated-values",
                                      "contentUrl": f"{GITHUB}/blob/main/data/oshiwambo_eval_v3.tsv"}]})
        out.append({"@context": "https://schema.org", "@type": "ScholarlyArticle",
                    "headline": "The Ongiini-Eval-OW Benchmark", "url": ARXIV,
                    "author": [{"@type": "Person", "name": "Sebastian Küpers"}], "datePublished": "2026-09"})
    else:
        out.append({"@context": "https://schema.org", "@type": "WebPage", "name": page["title"],
                    "url": url, "description": page["description"], "isPartOf": {"@type": "WebSite",
                                                                                  "name": BRAND, "url": SITE + "/"}})
    bc = breadcrumbs(page)
    if bc:
        out.append(bc)
    return out


# ── head block ───────────────────────────────────────────────────────

def block(page: dict, indent: str) -> str:
    url = absolute(page["url"])
    img = f"{SITE}/assets/og/{page['og']}.jpg"
    robots = page.get("robots", "index, follow, max-image-preview:large")
    lines = [
        "<!-- seo:start --> <!-- generated by scripts/build_site_seo.py — edit the PAGES table there -->",
        f"<title>{esc(page['title'])}</title>",
        f'<meta name="description" content="{esc(page["description"])}" />',
        f'<meta name="robots" content="{robots}" />',
        f'<link rel="canonical" href="{url}" />',
        f'<meta name="theme-color" content="#f5efe4" />',
        f'<link rel="icon" type="image/png" sizes="32x32" href="{SITE}/assets/favicon-32-v2.png" />',
        f'<link rel="icon" type="image/png" sizes="192x192" href="{SITE}/assets/ongiini-icon-192-v2.png" />',
        f'<link rel="apple-touch-icon" sizes="180x180" href="{SITE}/assets/apple-touch-icon.png" />',
        '<meta property="og:type" content="website" />',
        f'<meta property="og:site_name" content="{BRAND}" />',
        f'<meta property="og:title" content="{esc(page["title"])}" />',
        f'<meta property="og:description" content="{esc(page["description"])}" />',
        f'<meta property="og:url" content="{url}" />',
        f'<meta property="og:image" content="{img}" />',
        '<meta property="og:image:width" content="1200" />',
        '<meta property="og:image:height" content="630" />',
        f'<meta property="og:image:alt" content="{esc(page["og_alt"])}" />',
        '<meta property="og:locale" content="en_NA" />',
    ]
    if page.get("locale_alt"):
        lines.append(f'<meta property="og:locale:alternate" content="{page["locale_alt"]}" />')
    lines += [
        '<meta name="twitter:card" content="summary_large_image" />',
        f'<meta name="twitter:title" content="{esc(page["title"])}" />',
        f'<meta name="twitter:description" content="{esc(page["description"])}" />',
        f'<meta name="twitter:image" content="{img}" />',
    ]
    for obj in jsonld(page):
        lines.append('<script type="application/ld+json">' + json.dumps(obj, ensure_ascii=False) + "</script>")
    lines.append("<!-- seo:end -->")
    return "".join(f"{indent}{l}\n" for l in lines)


def apply(doc: str, page: dict) -> str:
    head_end = doc.index("</head>")
    head, rest = doc[:head_end], doc[head_end:]
    head = BLOCK_RE.sub("", head)
    for pat in OWNED:
        head = re.sub(r"[ \t]*" + pat + r"[ \t]*\n?", "", head, flags=re.S | re.I)
    m = re.search(r'([ \t]*)<meta\s+name="viewport"[^>]*>\n', head)
    if not m:
        raise SystemExit(f"{page['file']}: no viewport meta to anchor the block")
    head = head[:m.end()] + block(page, m.group(1)) + head[m.end():]
    return head + rest


# ── sitemap ──────────────────────────────────────────────────────────

def lastmod(path: Path) -> str:
    """Last commit date of the file; today if it has uncommitted changes."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(path)], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        return date.today().isoformat()
    out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    return out or date.today().isoformat()


def sitemap() -> str:
    rows = []
    for p in PAGES:
        if not p["url"].startswith("/"):
            continue                      # other hosts publish their own sitemap
        src = ROOT / p["template"] if p.get("template") else WEB / p["file"]
        rows.append(f"  <url>\n    <loc>{absolute(p['url'])}</loc>\n    <lastmod>{lastmod(src)}</lastmod>\n"
                    f"    <priority>{p['priority']}</priority>\n  </url>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + "\n".join(rows) + "\n</urlset>\n")


# ── rules ────────────────────────────────────────────────────────────

def problems(page: dict, doc: str) -> list[str]:
    errs = []
    t, d = page["title"], page["description"]
    if len(t) > 60:
        errs.append(f"title {len(t)} chars (> 60)")
    if not (t.endswith(BRAND) or (page.get("home") and t.startswith(BRAND))):
        errs.append("title does not end with 'Ongiini AI'")
    if not 120 <= len(d) <= 160:
        errs.append(f"description {len(d)} chars (want 120–160)")
    for word in (t, d):
        if re.search(r"\bhelper\b", word, re.I):
            errs.append("uses 'helper' (say 'AI assistant')")
    if not (WEB / "assets/og" / f"{page['og']}.jpg").exists():
        errs.append(f"missing og image assets/og/{page['og']}.jpg")
    body = doc[doc.index("<body"):]
    h1 = len(re.findall(r"<h1[\s>]", body))
    if h1 != 1:
        errs.append(f"{h1} <h1> elements (want 1)")
    for blob in re.findall(r'<script type="application/ld\+json">(.*?)</script>', doc, re.S):
        try:
            json.loads(blob)
        except ValueError as e:
            errs.append(f"invalid JSON-LD: {e}")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    failed = []
    targets = []
    for p in PAGES:
        targets.append((p, WEB / p["file"]))
        if p.get("template"):
            targets.append((p, ROOT / p["template"]))
    for page, path in targets:
        doc = path.read_text()
        new = apply(doc, page)
        if args.check:
            if new != doc:
                failed.append(f"{path.relative_to(ROOT)}: SEO block out of date — run scripts/build_site_seo.py")
            if path.suffix == ".html" and "template" not in str(path):
                failed += [f"{path.relative_to(ROOT)}: {e}" for e in problems(page, new)]
        elif new != doc:
            path.write_text(new)
            print(f"updated {path.relative_to(ROOT)}")
    sm = sitemap()
    robots = (WEB / "robots.txt").read_text()
    if args.check:
        locs = lambda x: sorted(re.findall(r"<loc>(.*?)</loc>", x))
        if locs((WEB / "sitemap.xml").read_text()) != locs(sm):
            failed.append("website/sitemap.xml lists other pages than PAGES — run scripts/build_site_seo.py")
        for p in PAGES:
            if p["url"].startswith("/") and p["url"] != "/" and f"Disallow: {p['url']}" in robots:
                failed.append(f"robots.txt blocks indexable page {p['url']}")
    elif (WEB / "sitemap.xml").read_text() != sm:
        (WEB / "sitemap.xml").write_text(sm)
        print("updated website/sitemap.xml")
    if failed:
        print("\n".join(failed))
        return 1
    if args.check:
        print(f"SEO OK — {len(PAGES)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
