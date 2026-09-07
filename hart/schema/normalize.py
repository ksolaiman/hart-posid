from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_legacy_value(value: Any) -> Any:
    """Conservatively replace legacy string nulls throughout a value."""
    if value == "null":
        return None
    if isinstance(value, Mapping):
        return {key: normalize_legacy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_legacy_value(item) for item in value]
    return value


def normalize_legacy_wearing(value: Any) -> list[dict[str, Any]]:
    """Convert arbitrary legacy clothing keys into stable item records."""
    normalized = normalize_legacy_value(value)
    if not isinstance(normalized, Mapping):
        raise ValueError("Legacy wearing annotations must be objects")
    return [
        {"item": item, "descriptors": descriptors}
        for item, descriptors in normalized.items()
    ]
