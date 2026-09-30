"""Unit tests for the PII sanitiser.

Covers the patterns that DO scrub (email, phone, credit-card, IBAN, ID), AND
the URL carve-out — URLs are public addresses and must not be touched,
even when their path contains digit sequences that look like a credit
card or National ID.
"""

from __future__ import annotations

import pytest

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


# ---------- phone numbers: mobiles go, landlines / service numbers stay ----------

@pytest.mark.parametrize("number", [
    "0812345678",
    "081 234 5678",
    "085-123-4567",
    "+264 81 234 5678",
    "+264812345678",
    "00264 81 234 5678",
    "264812345678",          # bare country code, as users type it
    "+49 151 23456789",      # foreign mobile (whitelisted pilot users)
    "+27 82 123 4567",
])
def test_mobile_numbers_redacted(number):
    out = sanitize(f"Please call me on {number} after five.")
    assert "[REDACTED:phone]" in out
    assert number not in out
    assert out.startswith("Please call me on ") and out.endswith(" after five.")


def test_international_mobile_no_longer_surfaces_as_card():
    """Before the phone pattern, long "+" numbers hit the card regex and
    left a stray "+" behind: "+[REDACTED:card]"."""
    out = sanitize("contact +264 81 234 5678 or +49 151 23456789")
    assert "[REDACTED:card]" not in out
    assert "+[REDACTED" not in out
    assert out.count("[REDACTED:phone]") == 2


@pytest.mark.parametrize("text", [
    "Call BIPA on 061 374 400 for company registration.",
    "The clinic line is +264 61 203 9111.",
    "In an emergency dial 10111.",
    "The meeting starts at 08:30 on 08-12-2026.",
    "Order number 1234567890 was shipped.",
])
def test_landlines_service_numbers_and_dates_untouched(text):
    assert sanitize(text) == text


def test_eleven_digit_id_starting_with_08_stays_an_id():
    """An 11-digit number must not be half-matched as a mobile."""
    out = sanitize("ID 08123456789 on the form")
    assert "[REDACTED:id]" in out
    assert "[REDACTED:phone]" not in out


def test_phone_inside_url_preserved():
    url = "https://wa.me/264812345678"
    assert sanitize(f"chat here {url}") == f"chat here {url}"


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
