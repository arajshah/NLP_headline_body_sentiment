from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b\S+@\S+\b")
HTML_ENT_RE = re.compile(r"&[a-z]+;")
WS_RE = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """
    Minimal cleaning intended for sentiment scoring:
    - lowercases
    - removes URLs/emails
    - unescapes common HTML entities (&amp; -> &), then strips remaining entity blobs
    - normalizes “smart quotes”
    - collapses whitespace
    """
    if not isinstance(text, str):
        return ""

    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = html.unescape(t)
    t = HTML_ENT_RE.sub(" ", t)
    t = (
        t.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    t = WS_RE.sub(" ", t).strip()
    return t


