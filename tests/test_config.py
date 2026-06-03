from pathlib import Path

from llm_drift.config import Config


def test_config_defaults_when_file_missing(tmp_path: Path):
    cfg = Config.load(tmp_path / "nope.yaml")
    assert cfg.store_path == ".llm-drift/baselines.db"
    assert cfg.drift_threshold == 0.15
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.alerts == [{"type": "stdout"}]


def test_config_loads_values_from_yaml(tmp_path: Path):
    f = tmp_path / "llm-drift.yaml"
    f.write_text(
        "store:\n"
        "  path: /custom/baselines.db\n"
        "thresholds:\n"
        "  drift_score: 0.30\n"
        "embedding:\n"
        "  model: my-model\n"
        "alerts:\n"
        "  - type: slack\n"
        "    webhook_url: https://hooks.slack.com/x\n"
    )
    cfg = Config.load(f)
    assert cfg.store_path == "/custom/baselines.db"
    assert cfg.drift_threshold == 0.30
    assert cfg.embedding_model == "my-model"
    assert cfg.alerts[0]["type"] == "slack"


def test_config_partial_yaml_uses_defaults(tmp_path: Path):
    f = tmp_path / "llm-drift.yaml"
    f.write_text("thresholds:\n  drift_score: 0.5\n")
    cfg = Config.load(f)
    assert cfg.drift_threshold == 0.5
    assert cfg.store_path == ".llm-drift/baselines.db"   # default
    assert cfg.alerts == [{"type": "stdout"}]            # default


def test_config_empty_yaml_uses_defaults(tmp_path: Path):
    f = tmp_path / "llm-drift.yaml"
    f.write_text("")
    cfg = Config.load(f)
    assert cfg.drift_threshold == 0.15
