import pytest
from pathlib import Path
from pydantic import ValidationError

from llm_drift.models import Probe, ProbeSuite


# ---------------------------------------------------------------------------
# Probe tests
# ---------------------------------------------------------------------------

def test_probe_requires_id_and_prompt():
    with pytest.raises(ValidationError):
        Probe()


def test_probe_requires_id():
    with pytest.raises(ValidationError):
        Probe(prompt="Say hello.")


def test_probe_requires_prompt():
    with pytest.raises(ValidationError):
        Probe(id="greet")


def test_probe_assertions_default_empty():
    probe = Probe(id="greet", prompt="Say hello.")
    assert probe.assertions == []


def test_probe_accepts_assertions():
    probe = Probe(id="greet", prompt="Say hello.", assertions=["output.min_length(1)"])
    assert probe.assertions == ["output.min_length(1)"]


# ---------------------------------------------------------------------------
# ProbeSuite tests
# ---------------------------------------------------------------------------

def _minimal_suite_data():
    return {
        "name": "my-suite",
        "model": "gpt-4o",
        "probes": [{"id": "p1", "prompt": "Say hello."}],
    }


def test_suite_requires_name_model_probes():
    with pytest.raises(ValidationError):
        ProbeSuite()


def test_suite_default_provider_is_openai():
    suite = ProbeSuite(**_minimal_suite_data())
    assert suite.provider == "openai"


def test_suite_provider_can_be_overridden():
    data = {**_minimal_suite_data(), "provider": "anthropic"}
    suite = ProbeSuite(**data)
    assert suite.provider == "anthropic"


def test_suite_probes_are_probe_instances():
    suite = ProbeSuite(**_minimal_suite_data())
    assert all(isinstance(p, Probe) for p in suite.probes)


def test_suite_rejects_duplicate_probe_ids():
    data = {
        "name": "dupe-suite",
        "model": "gpt-4o",
        "probes": [
            {"id": "same", "prompt": "a"},
            {"id": "same", "prompt": "b"},
        ],
    }
    with pytest.raises(ValidationError) as exc_info:
        ProbeSuite(**data)
    assert "unique" in str(exc_info.value)


# ---------------------------------------------------------------------------
# ProbeSuite.from_yaml tests
# ---------------------------------------------------------------------------

VALID_YAML = """\
name: invoice-suite
model: gpt-4o
provider: openai
probes:
  - id: extract-json
    prompt: "Extract fields from: Invoice #1, total $10"
    assertions:
      - "output.is_valid_json()"
  - id: tone-check
    prompt: "Reply professionally to: your service is bad"
"""

YAML_MISSING_MODEL = """\
name: invoice-suite
probes:
  - id: p1
    prompt: "Hello"
"""

YAML_EMPTY_PROBES = """\
name: invoice-suite
model: gpt-4o
probes: []
"""


def test_suite_from_yaml_valid(tmp_path: Path):
    f = tmp_path / "suite.yaml"
    f.write_text(VALID_YAML)

    suite = ProbeSuite.from_yaml(f)

    assert suite.name == "invoice-suite"
    assert suite.model == "gpt-4o"
    assert len(suite.probes) == 2
    assert suite.probes[0].id == "extract-json"
    assert suite.probes[1].id == "tone-check"


def test_suite_from_yaml_missing_model(tmp_path: Path):
    f = tmp_path / "suite.yaml"
    f.write_text(YAML_MISSING_MODEL)

    with pytest.raises(ValidationError) as exc_info:
        ProbeSuite.from_yaml(f)

    assert "model" in str(exc_info.value)


def test_suite_from_yaml_empty_probes(tmp_path: Path):
    f = tmp_path / "suite.yaml"
    f.write_text(YAML_EMPTY_PROBES)

    with pytest.raises(ValidationError):
        ProbeSuite.from_yaml(f)


def test_suite_from_yaml_file_not_found():
    with pytest.raises(FileNotFoundError):
        ProbeSuite.from_yaml("/nonexistent/path/suite.yaml")
