import json
import re
from typing import Any, Optional

VALID_LABELS = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}


def normalize_label(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.strip().lower()
    if lowered in VALID_LABELS:
        return VALID_LABELS[lowered]

    found = re.search(r"\b(positive|negative|neutral)\b", lowered)
    if found:
        return VALID_LABELS[found.group(1)]
    return None


def parse_structured_output(raw: str) -> tuple[Optional[str], bool]:
    try:
        data: Any = json.loads(raw)
        if not isinstance(data, dict):
            return None, False
        sentiment = data.get("sentiment")
        if not isinstance(sentiment, str):
            return None, False
        label = normalize_label(sentiment)
        return label, label is not None
    except json.JSONDecodeError:
        return None, False


def compliance_for_strategy(strategy_name: str, output: str) -> tuple[Optional[str], bool]:
    if strategy_name == "Structured":
        return parse_structured_output(output)

    label = normalize_label(output)
    return label, label is not None
