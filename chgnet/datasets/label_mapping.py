"""SDD raw string labels → internal taxonomy and class indices.

**paper-specified**: multi-class heterogeneous agents on SDD (high level).

**engineering assumption**: alias table below, collapse rules from YAML, lexicographic class index order,
and default ``collapse_other_as`` for ``collapsed_3`` (see ``data/label_mapping.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

# Normalized raw token → full internal class (before collapse).
# Keys must be lowercase stripped tokens (quotes removed before lookup).
_RAW_ALIASES: dict[str, str] = {
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "people": "pedestrian",
    "walker": "pedestrian",
    "biker": "cyclist",
    "cyclist": "cyclist",
    "bike": "cyclist",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "skater": "skater",
    "skateboard": "skater",
    "skateboarder": "skater",
    "car": "vehicle",
    "cart": "vehicle",
    "bus": "vehicle",
    "vehicle": "vehicle",
    "truck": "vehicle",
    "van": "vehicle",
}

_FULL_VOCAB_ORDER: tuple[str, ...] = ("cyclist", "other", "pedestrian", "skater", "vehicle")
_COLLAPSED_VOCAB_ORDER: tuple[str, ...] = ("cyclist", "pedestrian", "vehicle")


def normalize_raw_label(raw: str) -> str:
    """Lowercase, trim, strip wrapping quotes, collapse internal whitespace."""
    s = raw.strip().lower()
    s = s.strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class LabelMapper:
    """Maps raw SDD labels to internal strings and integer class indices."""

    taxonomy_mode: str
    unknown_internal_class: str
    collapse_skater_as: str
    collapse_other_as: str
    vocabulary: tuple[str, ...]

    def raw_to_full_internal(self, raw: str) -> str:
        key = normalize_raw_label(raw)
        if key == "" or key == "unknown":
            return self.unknown_internal_class
        return _RAW_ALIASES.get(key, self.unknown_internal_class)

    def full_internal_to_model_class(self, internal_full: str) -> str:
        if self.taxonomy_mode == "full":
            if internal_full not in _FULL_VOCAB_ORDER:
                return self.unknown_internal_class
            return internal_full
        if self.taxonomy_mode == "collapsed_3":
            if internal_full == "skater":
                return self.collapse_skater_as
            if internal_full == "other":
                return self.collapse_other_as
            if internal_full in ("pedestrian", "cyclist", "vehicle"):
                return internal_full
            return self.collapse_other_as
        raise ValueError(f"Unknown labels.taxonomy_mode: {self.taxonomy_mode!r}")

    def model_class_to_index(self, model_class: str) -> int:
        if model_class not in self.vocabulary:
            raise KeyError(f"Class {model_class!r} not in vocabulary {self.vocabulary}")
        return self.vocabulary.index(model_class)

    def map_raw(self, raw: str) -> tuple[str, str, int]:
        """Return ``(full_internal, model_class, class_index)``."""
        full = self.raw_to_full_internal(raw)
        model_c = self.full_internal_to_model_class(full)
        idx = self.model_class_to_index(model_c)
        return full, model_c, idx


def _vocabulary_for_mode(cfg_labels: Mapping[str, Any]) -> tuple[str, ...]:
    mode = cfg_labels["taxonomy_mode"]
    if mode == "full":
        return _FULL_VOCAB_ORDER
    if mode == "collapsed_3":
        return _COLLAPSED_VOCAB_ORDER
    raise ValueError(f"Unknown labels.taxonomy_mode: {mode!r}")


def build_label_mapper_from_config(cfg: Mapping[str, Any]) -> LabelMapper:
    """Build mapper from merged config dict (expects ``cfg['labels']``)."""
    labels = cfg["labels"]
    taxonomy_mode = labels["taxonomy_mode"]
    unknown = labels["unknown_internal_class"]
    collapse_skater = labels.get("collapse_skater_as", "cyclist")
    collapse_other = labels.get("collapse_other_as", "pedestrian")
    vocab = _vocabulary_for_mode(labels)

    if unknown not in _FULL_VOCAB_ORDER:
        raise ValueError(f"unknown_internal_class {unknown!r} must be a full-taxonomy class.")
    if taxonomy_mode == "collapsed_3":
        if collapse_skater not in _COLLAPSED_VOCAB_ORDER:
            raise ValueError(
                f"collapse_skater_as {collapse_skater!r} must be one of {_COLLAPSED_VOCAB_ORDER}."
            )
        if collapse_other not in _COLLAPSED_VOCAB_ORDER:
            raise ValueError(
                f"collapse_other_as {collapse_other!r} must be one of {_COLLAPSED_VOCAB_ORDER}."
            )
    return LabelMapper(
        taxonomy_mode=taxonomy_mode,
        unknown_internal_class=unknown,
        collapse_skater_as=collapse_skater,
        collapse_other_as=collapse_other,
        vocabulary=vocab,
    )


def vocabulary_dict(mapper: LabelMapper) -> dict[str, int]:
    """Deterministic string → index (for JSON metadata)."""
    return {name: i for i, name in enumerate(mapper.vocabulary)}
