"""Unit tests for the PII sanitiser.

Covers the patterns that DO scrub (email, credit-card, IBAN, ID), AND
the URL carve-out — URLs are public addresses and must not be touched,
even when their path contains digit sequences that look like a credit
card or National ID.
"""

from __future__ import annotations

from ongiini.pii import sanitize, sanitize_message


# ---------- positive: things that SHOULD be redacted ----------

def test_email_redacted():
    out = sanitize("Reach me at hello@example.com please.")
    assert "[REDACTED:email]" in out
    assert "hello@example.com" not in out


def test_credit_card_with_spaces_redacted():
    out = sanitize("My card is 4111 1111 1111 1111 thanks.")
    assert "[REDACTED:card]" in out
    assert "4111" not in out


def test_credit_card_continuous_digits_redacted():
    out = sanitize("Number 4111111111111111 fyi.")
    assert "[REDACTED:card]" in out


def test_iban_redacted():
    out = sanitize("IBAN NA47 0011 1234 5678 9012 — pay there")
    # IBAN regex matches the country code + check digits + alphanumeric
    # block; verify that SOMETHING got redacted.
    assert "[REDACTED:" in out


def test_eleven_digit_national_id_redacted():
    out = sanitize("My ID is 12345678901.")
    assert "[REDACTED:id]" in out
    assert "12345678901" not in out


# ---------- v1.6.1 URL carve-out: URLs must be preserved ----------

def test_facebook_video_url_with_long_digit_id_preserved():
    """Production transcript turned ``.../the-k/1234567890123456/`` into
    ``.../the-k/[REDACTED:card]`` because Facebook video IDs are 15-16
    digits and matched the credit-card regex. URLs must be peeled out
    before scrubbing."""
    url = "https://www.facebook.com/skateaid/videos/the-new-sun-sail-is-finally-set-up-at-our-skate-park-in-windhoeknamibia-so-the-k/1234567890123456/"
    out = sanitize(f"— source: {url}")
    assert "[REDACTED:" not in out
    assert url in out


def test_url_with_eleven_digit_path_preserved():
    """An 11-digit path component would otherwise match the National-ID
    pattern. Common in legacy news article slugs."""
    url = "https://www.namibian.com.na/article/12345678901"
    out = sanitize(f"see {url}")
    assert "[REDACTED:" not in out
    assert url in out


def test_url_with_query_string_digits_preserved():
    url = "https://example.com/page?id=4111111111111111&utm=track"
    out = sanitize(url)
    assert "[REDACTED:" not in out
    assert url in out


def test_http_and_https_both_preserved():
    text = "old http://example.com/4111111111111111 and https://example.com/4111111111111111"
    out = sanitize(text)
    assert "[REDACTED:" not in out


# ---------- mixed cases ----------

def test_credit_card_outside_url_still_redacted_when_url_in_text():
    """The URL carve-out must not let credit cards OUTSIDE URLs slip
    through. Defence in depth."""
    text = "see https://example.com/12345678901234 — my card is 4111-1111-1111-1111"
    out = sanitize(text)
    assert "https://example.com/12345678901234" in out
    assert "[REDACTED:card]" in out
    assert "4111-1111-1111-1111" not in out


def test_email_outside_url_still_redacted_when_url_in_text():
    text = "more at https://example.com/page or hello@example.com"
    out = sanitize(text)
    assert "https://example.com/page" in out
    assert "[REDACTED:email]" in out


def test_multiple_urls_in_one_text_all_preserved():
    text = (
        "— source: https://www.namibian.com.na/article/12345678901\n"
        "— source: https://www.facebook.com/videos/4111111111111111/"
    )
    out = sanitize(text)
    assert "[REDACTED:" not in out
    assert "https://www.namibian.com.na/article/12345678901" in out
    assert "https://www.facebook.com/videos/4111111111111111/" in out


# ---------- edge cases ----------

def test_empty_string_returns_empty():
    assert sanitize("") == ""


def test_text_without_pii_returned_unchanged():
    text = "Just a normal sentence about Windhoek with no sensitive bits."
    assert sanitize(text) == text


def test_sanitize_message_redacts_content_field():
    msg = {"role": "user", "content": "my email is hello@example.com"}
    out = sanitize_message(msg)
    assert out["role"] == "user"
    assert "[REDACTED:email]" in out["content"]


def test_sanitize_message_passes_through_non_string_content():
    """Image-bearing messages have list content. The sanitiser leaves
    non-string content unchanged."""
    msg = {"role": "user", "content": [{"type": "text", "text": "x"}]}
    out = sanitize_message(msg)
    assert out["content"] == [{"type": "text", "text": "x"}]
