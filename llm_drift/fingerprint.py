from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Literal, Protocol


class EmbeddingModel(Protocol):
    def encode(self, text: str) -> List[float]:
        ...


class SentenceTransformerModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install llm-drift[sentence-transformers]"
            )
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()


@dataclass
class Fingerprint:
    embedding: List[float]
    token_count: int
    format: Literal["json", "markdown", "plain"]
    assertion_results: List[dict] = field(default_factory=list)
    raw_output: str = ""


def _detect_format(text: str) -> Literal["json", "markdown", "plain"]:
    stripped = text.strip()
    try:
        json.loads(stripped)
        return "json"
    except (json.JSONDecodeError, ValueError):
        pass
    if re.search(r"(^#{1,4}\s|^\s*[-*]\s|\*\*|__)", stripped, re.MULTILINE):
        return "markdown"
    return "plain"


def fingerprint(text: str, model: EmbeddingModel) -> Fingerprint:
    return Fingerprint(
        embedding=list(model.encode(text)),
        token_count=max(len(text.split()), 1),
        format=_detect_format(text),
        raw_output=text,
    )
