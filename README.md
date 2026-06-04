# llm-drift

Detect when your LLM's behavior has silently changed over time.

LLM providers (OpenAI, Anthropic, Google) frequently update models under the same version string — same name, different behavior. `llm-drift` gives you an automated, statistical signal when this happens, before your users notice.

---

## The Problem

When you rely on a named model like `gpt-4o` or `claude-sonnet-4-6`, you implicitly assume stable behavior. In practice:

- Providers silently patch models with no changelog
- Output tone, format, verbosity, or reasoning can shift overnight
- Regressions are discovered by users, not developers

> A team running GPT-4 in production discovered their JSON extraction pipeline started failing — the model stopped wrapping outputs in code fences after a silent update. They only found out 3 days later via a customer complaint.

---

## How It Works

`llm-drift` runs a **probe suite** — a set of canonical prompts with expected behavior — on a schedule. Each run produces output **fingerprints** (embeddings + structural stats). Fingerprints are compared against a stored **baseline** to compute a **drift score** between `0.0` (identical) and `1.0` (completely different).

```
Probe Suite → LLM → Fingerprinter → Drift Scorer → Alert
                                         ↑
                                      Baseline (SQLite)
```

The drift score is a weighted combination of three signals:

| Signal | Method | Default Weight |
|---|---|---|
| Semantic | Cosine distance of output embeddings | 0.50 |
| Structural | Format change (JSON/markdown/plain) + token length ratio | 0.25 |
| Assertion regression | % of assertions newly failing vs. baseline | 0.25 |

---

## Installation

> **Note:** not yet published to PyPI. Until then, install from GitHub.

```bash
pip install git+https://github.com/beingaskar/llm-drift.git
```

With provider support and local embeddings:

```bash
# OpenAI + local embeddings (no embedding API key needed)
pip install "llm-drift[openai,sentence-transformers] @ git+https://github.com/beingaskar/llm-drift.git"

# Anthropic instead
pip install "llm-drift[anthropic,sentence-transformers] @ git+https://github.com/beingaskar/llm-drift.git"
```

Once published to PyPI, this becomes simply:

```bash
pip install "llm-drift[openai,sentence-transformers]"
```

Python 3.9+ required.

---

## Quickstart

### 1. Scaffold config and an example probe

```bash
llm-drift init
```

This creates `llm-drift.yaml` and `probes/example.yaml` in your project.

### 2. Define your probe suite

`llm-drift init` generates `probes/example.yaml` with 6 probes out of the box, each targeting a different failure mode:

```yaml
name: example-suite
model: gpt-4o
provider: openai
probes:
  # Basic sanity — model should always respond
  - id: hello
    prompt: "Say hello in one sentence."
    assertions:
      - "output.min_length(5)"
      - "output.max_length(300)"

  # JSON format stability — did the model stop returning structured output?
  - id: extract-json
    prompt: >
      Extract the fields from this invoice and return as JSON:
      'Invoice #4821, customer: Acme Corp, total: $149.99, due: 2026-07-15'
    assertions:
      - "output.is_valid_json()"
      - "output.contains(\"4821\")"
      - "output.contains(\"149.99\")"

  # Instruction following — model should respect explicit format rules
  - id: bullet-list
    prompt: "List 3 benefits of unit testing. Use a bullet point for each. No intro sentence."
    assertions:
      - "output.min_length(30)"
      - "output.max_length(600)"

  # Tone consistency — professional register should stay stable
  - id: professional-tone
    prompt: >
      A customer wrote: 'Your product is broken and I want a refund NOW.'
      Write a professional one-paragraph response acknowledging their frustration.
    assertions:
      - "output.min_length(50)"
      - "output.not_contains(\"I cannot\")"
      - "output.not_contains(\"As an AI\")"

  # Summarisation length — verbosity drift is common after model updates
  - id: summarise-short
    prompt: >
      Summarise this in exactly 2 sentences: 'Machine learning is a branch of
      artificial intelligence that enables systems to learn from data...'
    assertions:
      - "output.min_length(40)"
      - "output.max_length(400)"

  # Refusal stability — model should answer this, not refuse
  - id: no-refusal
    prompt: "What is the capital of France?"
    assertions:
      - "output.contains(\"Paris\")"
      - "output.not_contains(\"cannot\")"
```

| Probe | What it catches |
|---|---|
| `hello` | Model stops responding or gives empty output |
| `extract-json` | Model stops returning structured JSON |
| `bullet-list` | Model stops following explicit format instructions |
| `professional-tone` | Tone shifts; model starts refusing with "As an AI..." |
| `summarise-short` | Verbosity drift — responses get much longer or shorter |
| `no-refusal` | Model starts refusing simple factual questions |

### 3. Capture a baseline

```bash
llm-drift baseline --suite probes/example.yaml
```

### 4. Run drift detection

```bash
llm-drift run --suite probes/example.yaml
# Suite: example-suite  score: 0.031  [OK]

# Exit with code 1 if drift detected — useful in CI
llm-drift run --suite probes/example.yaml --fail-on-drift
```

### 5. View history

```bash
llm-drift report --suite example-suite
```

```
Run ID                                  Date                        Score  Status
-------------------------------------------------------------------------------------
3f2a1b...                               2026-06-03T09:00:00+00:00   0.031  OK
1c4d9e...                               2026-06-02T09:00:00+00:00   0.028  OK
```

---

## Python API

```python
import asyncio
from llm_drift import Probe, ProbeSuite
from llm_drift.adapters import OpenAIAdapter
from llm_drift.fingerprint import SentenceTransformerModel
from llm_drift.store import SQLiteStore
from llm_drift.runner import SuiteRunner

suite = ProbeSuite(
    name="invoice-extractor",
    model="gpt-4o",
    probes=[
        Probe(
            id="extract-json",
            prompt="Extract invoice fields from: 'Invoice #1234, total $99.00'",
            assertions=["output.is_valid_json()"],
        )
    ],
)

async def main():
    runner = SuiteRunner(
        suite=suite,
        adapter=OpenAIAdapter(model="gpt-4o"),
        store=SQLiteStore(),
        embedding_model=SentenceTransformerModel(),
    )

    # First time: capture a baseline
    await runner.capture_baseline()

    # On subsequent runs: detect drift
    result = await runner.run()
    print(result.drift_score)   # 0.0 = identical, 1.0 = completely different
    print(result.drifted)       # True if score > threshold (default 0.15)
    print(result.drift_result.report())

asyncio.run(main())
```

---

## Built-in Assertions

Assertions are DSL strings evaluated against the LLM output:

| Assertion | Example |
|---|---|
| Valid JSON | `output.is_valid_json()` |
| Contains substring | `output.contains("invoice")` |
| Does not contain | `output.not_contains("error")` |
| Min length | `output.min_length(50)` |
| Max length | `output.max_length(500)` |

---

## Alerts

Configure alert channels in `llm-drift.yaml`:

```yaml
thresholds:
  drift_score: 0.15

alerts:
  - type: stdout
  - type: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
  - type: webhook
    url: https://your-endpoint.com/llm-drift
```

Or wire up alerts in code:

```python
from llm_drift.alerts import AlertDispatcher, SlackAlertBackend

dispatcher = AlertDispatcher(
    backends=[SlackAlertBackend(webhook_url="https://hooks.slack.com/...")],
    threshold=0.15,
)
await dispatcher.dispatch(result)
```

---

## CI Integration

Add drift detection to your pipeline:

```yaml
# .github/workflows/drift.yml
- name: Check for LLM drift
  run: llm-drift run --suite probes/suite.yaml --fail-on-drift
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## CLI Reference

```
llm-drift init                                    Scaffold config and example probe
llm-drift baseline --suite <path>                 Capture a baseline
llm-drift baseline --suite <path> --no-strict     Capture even if assertions fail
llm-drift run --suite <path>                      Run drift detection (+ fires alerts)
llm-drift run --suite <path> --fail-on-drift      Exit 1 if drifted (CI)
llm-drift report --suite <name|path>              Show drift history table
llm-drift diff --suite <name|path>                Show baseline vs. latest run, per probe
llm-drift diff --suite <name|path> --run <id>     Diff against a specific run
```

`baseline`, `run`, `report`, and `diff` accept `--config <path>` (defaults to
`./llm-drift.yaml`). `report` and `diff` accept either the suite **name** or the same
YAML **path** you pass to `baseline`/`run`.

### Notes

- **Determinism:** adapters call the model with `temperature=0` so sampling variance
  doesn't masquerade as drift. Some residual non-determinism is inherent to LLMs.
- **Baseline integrity:** `baseline` refuses to store a snapshot whose assertions
  already fail (use `--no-strict` to override). Probes are matched to the baseline by
  `id`, so reordering, adding, or removing a probe is reported as an error rather than
  silently mis-compared.
- **Alerts:** `run` dispatches the alert channels configured under `alerts:` in
  `llm-drift.yaml` whenever the drift score exceeds the threshold.

---

## Architecture

```
llm_drift/
├── models.py        Probe, ProbeSuite (Pydantic, unique-id validation)
├── config.py        Config loader for llm-drift.yaml
├── adapters.py      ProviderAdapter protocol, OpenAIAdapter, AnthropicAdapter
├── fingerprint.py   Fingerprint dataclass, format detection, EmbeddingModel protocol
├── assertions.py    AssertionRunner, DSL evaluation
├── store.py         BaselineStore ABC, SQLiteStore (baselines + run outputs)
├── scorer.py        DriftScorer, DriftResult, probe_id-keyed matching
├── runner.py        SuiteRunner, capture_baseline(), run()
├── alerts.py        AlertDispatcher, Stdout/Slack/Webhook backends, build_dispatcher
└── cli.py           Click CLI entry point
```

---

## Development

```bash
git clone https://github.com/beingaskar/llm-drift.git
cd llm-drift
pip install -e ".[dev]"
pytest
```

---

## Roadmap

- **v0.1** — Core probe model, OpenAI + Anthropic adapters, SQLite store, CLI
- **v0.2** — LiteLLM adapter (100+ models), Ollama, S3 store
- **v0.3** — GitHub Actions workflow, HTML reports, drift dashboard
- **v1.0** — PostgreSQL store, custom scorer plugin API, full docs site

---

## License

MIT
