"""Entity extraction for AgentDojo task authorization.

The extractor is intentionally lightweight and deterministic. It is used as
evidence, not as an oracle: fair mode only sees user task text, tool arguments,
and prior tool outputs observed by the firewall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_RE = re.compile(r"https?://[^\s)\]}>'\"]+|www\.[^\s)\]}>'\"]+")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
MONEY_RE = re.compile(r"(?i)(?:\$|鈧瑋eur|usd|gbp)?\s*\b\d+(?:\.\d{1,2})?\b")
DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b", re.I
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", re.I)
CHANNEL_RE = re.compile(r"(?i)(?:#|channel\s+)([a-z0-9_-]{2,40})")
FILE_RE = re.compile(r"(?i)\b[\w .-]+\.(?:txt|md|pdf|docx|xlsx|csv|json|py)\b")
QUOTED_RE = re.compile(r"\"([^\"]{3,80})\"|(?<!\w)'([^']{3,80})'(?!\w)")
BOOKING_RE = re.compile(r"(?i)\b(?:reserve|book|booking|hotel|restaurant)\s+([A-Z][A-Za-z0-9 _.-]{2,60})")

ENTITY_KEYS = {
    "email",
    "url",
    "iban",
    "amount",
    "recipient",
    "hotel",
    "restaurant",
    "company",
    "file_id",
    "filename",
    "date",
    "time",
    "channel",
    "user",
}

PLURAL_KEY_ALIASES = {
    "hotel_names": "hotel",
    "restaurant_names": "restaurant",
    "company_names": "company",
    "recipient_names": "recipient",
    "user_names": "user",
    "channel_names": "channel",
    "file_names": "filename",
}


@dataclass(slots=True)
class EntitySet:
    values: dict[str, set[str]] = field(default_factory=dict)

    def add(self, kind: str, value: Any) -> None:
        text = normalize_entity(value)
        if not text:
            return
        self.values.setdefault(kind, set()).add(text)

    def flattened(self) -> set[str]:
        out: set[str] = set()
        for values in self.values.values():
            out.update(values)
        return out

    def values_for(self, kind: str) -> set[str]:
        return set(self.values.get(kind, set()))

    def overlaps(self, other: "EntitySet") -> bool:
        return bool(self.flattened() & other.flattened())

    def matches_text(self, text: Any) -> bool:
        haystack = normalize_text(text)
        for value in self.flattened():
            needle = normalize_text(value)
            if not needle:
                continue
            if needle in haystack:
                return True
        return False

    def as_dict(self) -> dict[str, list[str]]:
        return {key: sorted(values) for key, values in sorted(self.values.items())}


def extract_entities(value: Any) -> EntitySet:
    text = stringify(value)
    entities = EntitySet()
    for item in EMAIL_RE.findall(text):
        entities.add("email", item)
        entities.add("recipient", item)
    for item in URL_RE.findall(text):
        entities.add("url", item)
    for item in IBAN_RE.findall(text):
        entities.add("iban", item)
        entities.add("recipient", item)
    for item in DATE_RE.findall(text):
        entities.add("date", item)
    for item in TIME_RE.findall(text):
        entities.add("time", item)
    for item in CHANNEL_RE.findall(text):
        entities.add("channel", item)
    for item in FILE_RE.findall(text):
        entities.add("filename", item)
    for match in QUOTED_RE.findall(text):
        item = next((part for part in match if part), "")
        entities.add("quoted", item)
        entities.add("hotel", item)
        entities.add("restaurant", item)
    for item in BOOKING_RE.findall(text):
        cleaned = re.split(r"(?i)\s+(?:if|from|on|for|and|with|because)\b", item.strip(), maxsplit=1)[0]
        entities.add("hotel", cleaned)
        entities.add("restaurant", cleaned)

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            canonical_key = PLURAL_KEY_ALIASES.get(key_text, key_text)
            if canonical_key in ENTITY_KEYS:
                entities.add(canonical_key, item)
            elif canonical_key.endswith("_name") and canonical_key[:-5] in ENTITY_KEYS:
                entities.add(canonical_key[:-5], item)
            elif canonical_key.endswith("_names") and canonical_key[:-6] in ENTITY_KEYS:
                entities.add(canonical_key[:-6], item)
            elif key_text in {"to", "user_email"}:
                entities.add("email", item)
                entities.add("recipient", item)
            elif key_text in {"body", "message", "subject", "content", "query"}:
                nested = extract_entities(item)
                for nested_key, nested_values in nested.values.items():
                    for nested_value in nested_values:
                        entities.add(nested_key, nested_value)
            elif isinstance(item, (dict, list, tuple, set)):
                nested = extract_entities(item)
                for nested_key, nested_values in nested.values.items():
                    for nested_value in nested_values:
                        entities.add(nested_key, nested_value)
    return entities


def normalize_entity(value: Any) -> str:
    text = str(value or "").strip().strip("'\"").rstrip(".,;:)")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def normalize_text(value: Any) -> str:
    text = normalize_entity(value)
    text = re.sub(r"[^a-z0-9@._-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k}={stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(stringify(item) for item in value)
    return str(value or "")

