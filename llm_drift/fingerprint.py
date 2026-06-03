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


def build_embedding_model(name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Factory for embedding models named in config.

    Currently only local sentence-transformers models are supported. Remote
    embedding providers (e.g. ``openai/text-embedding-3-small``) are planned but
    not yet implemented — fail loudly rather than silently falling back.
    """
    if name.startswith("openai/"):
        raise NotImplementedError(
            f"Embedding model {name!r} is not supported yet. "
            "Only local sentence-transformers models work today (e.g. 'all-MiniLM-L6-v2')."
        )
    return SentenceTransformerModel(name)


@dataclass
class Fingerprint:
    embedding: List[float]
    token_count: int
    format: Literal["json", "markdown", "plain"]
    assertion_results: List[dict] = field(default_factory=list)
    raw_output: str = ""
    probe_id: str = ""


def _detect_format(text: str) -> Literal["json", "markdown", "plain"]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        # Only structured payloads count as JSON. Bare scalars like "42",
        # "true", or "null" also parse but are not meaningfully "JSON output".
        if isinstance(parsed, (dict, list)):
            return "json"
    except (json.JSONDecodeError, ValueError):
        pass
    if re.search(r"(^#{1,4}\s|^\s*[-*]\s|\*\*|__)", stripped, re.MULTILINE):
        return "markdown"
    return "plain"


def fingerprint(text: str, model: EmbeddingModel, probe_id: str = "") -> Fingerprint:
    text = text or ""
    return Fingerprint(
        embedding=list(model.encode(text)),
        token_count=max(len(text.split()), 1),
        format=_detect_format(text),
        raw_output=text,
        probe_id=probe_id,
    )
