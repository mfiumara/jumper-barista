"""A small deterministic intent parser for the café order, plus transcript normalisation.

``parse_intent(transcript)`` returns ``{"intent": "order_drink", "drink": <id>, "matched":
<phrase>}`` for the first drink phrase found (the most specific drink wins), or
``{"intent": "unknown", "drink": "unknown", "matched": None}``. No model, no randomness.
"""

from __future__ import annotations

import re
import unicodedata

_NUMBER_WORDS = {"sixty": "60"}

# Order matters: the first entry whose pattern matches wins, so specific drinks come
# before generic ones ("filter coffee" before "coffee").
DRINKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("v60_pour_over", (
        r"\b(v|vee) ?60\b",
        r"\bpour ?over\b",
        r"\bhand ?drip\b",
        r"\bfilter coffee\b",
        r"\bdrip coffee\b",
    )),
    ("cappuccino", (r"\bcappuccino\b", r"\bcapuccino\b")),
    ("latte", (r"\blatte\b", r"\bcaffe latte\b")),
    ("espresso", (r"\bespresso\b", r"\bexpresso\b")),
)
_ORDER_CUES = (
    r"\bcould i (get|have)\b", r"\bcan i (get|have)\b", r"\bi(d| would) like\b",
    r"\bplease\b", r"\bone\b", r"\bi ll (have|take)\b", r"\bmake me\b", r"\bgive me\b",
)


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, unify 'pour-over'/'V-60', collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    text = text.replace("\u2014", " ").replace("\u2013", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text).replace("'", " ")
    text = re.sub(r"\s+", " ", text).strip()
    for word, digits in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digits, text)
    text = re.sub(r"\bv 60\b", "v60", text)
    text = re.sub(r"\bpourover\b", "pour over", text)
    return text


def parse_intent(transcript: str) -> dict:
    text = normalise(transcript)
    for drink, patterns in DRINKS:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    "intent": "order_drink",
                    "drink": drink,
                    "matched": match.group(0),
                    "order_cue": any(re.search(cue, text) for cue in _ORDER_CUES),
                }
    return {"intent": "unknown", "drink": "unknown", "matched": None, "order_cue": False}
