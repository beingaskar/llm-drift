import hashlib
import math
import random
from typing import List

from llm_drift.fingerprint import Fingerprint, fingerprint, _detect_format


class MockEmbeddingModel:
    """Deterministic mock: same text → same 384-float vector."""

    def encode(self, text: str) -> List[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.gauss(0, 1) for _ in range(384)]


_MODEL = MockEmbeddingModel()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = lambda v: math.sqrt(sum(x * x for x in v))
    return dot / (mag(a) * mag(b))


# ---------------------------------------------------------------------------
# Fingerprint shape and fields
# ---------------------------------------------------------------------------

def test_fingerprint_embedding_shape():
    fp = fingerprint("Hello world", _MODEL)
    assert len(fp.embedding) == 384
    assert all(isinstance(v, float) for v in fp.embedding)


def test_fingerprint_token_count_nonzero():
    fp = fingerprint("Hi", _MODEL)
    assert fp.token_count > 0


def test_fingerprint_raw_output_stored():
    text = "Some output text"
    fp = fingerprint(text, _MODEL)
    assert fp.raw_output == text


def test_fingerprint_assertions_default_empty():
    fp = fingerprint("test", _MODEL)
    assert fp.assertion_results == []


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def test_fingerprint_detects_json_format():
    fp = fingerprint('{"key": "value", "n": 1}', _MODEL)
    assert fp.format == "json"


def test_fingerprint_detects_json_array():
    fp = fingerprint('[1, 2, 3]', _MODEL)
    assert fp.format == "json"


def test_fingerprint_detects_markdown_format():
    fp = fingerprint("## Heading\n- item one\n- item two", _MODEL)
    assert fp.format == "markdown"


def test_fingerprint_detects_plain_format():
    fp = fingerprint("Just a plain sentence with no special formatting.", _MODEL)
    assert fp.format == "plain"


# ---------------------------------------------------------------------------
# Semantic stability
# ---------------------------------------------------------------------------

def test_fingerprint_identical_texts_close_embeddings():
    text = "The quick brown fox jumps over the lazy dog."
    fp1 = fingerprint(text, _MODEL)
    fp2 = fingerprint(text, _MODEL)
    sim = _cosine_similarity(fp1.embedding, fp2.embedding)
    assert sim > 0.999
