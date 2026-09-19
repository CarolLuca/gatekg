# GateKG

**Gate Before You Act: Structural Tool-Use Control for Mutable Agent Memory**

Accepted to Findings of EACL 2027.

GateKG is a Python framework for language-model experiments with knowledge-graph memory. It supports direct and recursive inference, read-only and writable graph agents, learned policy selection, and tool-call gating.

For each question, a selector uses the question and baseline graph-health measurements to choose read-only or read-write access. A second model scores each proposed tool call using the baseline measurements and the calls attempted so far. Reads can retrieve evidence from the cached base graph and the question’s temporary work graph. Writes are scoped to the work graph, which is discarded after the question.

This repository provides the experiment runners, training pipeline and analysis tools. The [paper workflow](docs/paper.md) maps the experimental settings to commands.

## Quick start

Requires Python 3.11 or newer. Run these commands from the repository root:

```bash
python -m pip install -e ".[adaptive,analysis,data,dev]"
cp .env.example .env
gatekg list
gatekg run examples/qa.jsonl llm --judge exact --dry-run
```

The dry run should report `"dataset": "qa"`, `"entries": 2`, and `"runner": "llm"`. It runs on CPU without a model service. For local inference, the model weights and context cache must fit in available RAM or VRAM. Local graph runs also need disk for Neo4j and an embedding model. The paper’s local hardware included GPUs with 48 GB and 16 GB of VRAM.

The default provider is Ollama. Start Ollama and pull the configured models before running inference:

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text:latest
gatekg run examples/qa.jsonl llm --judge exact
```

Graph-backed runners also need Neo4j. Set its credentials in `.env`, then run:

```bash
docker compose up -d neo4j
gatekg doctor --runner read_only_rlm
gatekg run examples/qa.jsonl read_only_rlm --judge exact
```

Graph extraction uses the configured KG model. Google reranking is used when Google credentials are configured; otherwise local graph runs use rank fusion. Use a dedicated database for experiments. Recursive runners execute model-generated Python locally. The learned gate controls graph tools. Use process or container isolation to restrict filesystem, network, and credential access.

## Configuration

Use `gatekg --config configs/default.yaml ...` or set `GATEKG_CONFIG`.
Model names, endpoints, context limits, and runtime directories are configurable.
See [configuration](docs/configuration.md) for provider profiles and environment overrides.

Input data uses JSONL records with `id`, `question`, `answer`, and `context`.
Context may be a string or a list of strings. Field names and chunking can be changed through CLI options.

```bash
gatekg run path/to/questions.jsonl llm --question-field query --answer-field expected
gatekg batch configs/experiments.example.yaml --dry-run
gatekg batch configs/experiments.example.yaml
```

Each run saves a result report, a log, and an input manifest. Manifests snapshot the evaluated entries, so training does not need the original dataset file. These outputs can contain sensitive inputs and responses. Keep them private.

## Train a policy

Install the `adaptive` extra and collect runs with the graph-backed baselines. Then:

```bash
gatekg-build-dataset --log-root logs
gatekg-train --dataset data/gatekg/runs.jsonl
gatekg run examples/qa.jsonl gatekg --artifact-dir artifacts/gatekg/shared --judge exact
```

`gatekg-advance` combines dataset construction and training. Add
`--gate-label-source trajectory_state` to use Google-labeled tool trajectories.
`gatekg-interpret` inspects datasets and run traces. See [training](docs/training.md)
for details and artifact requirements.

## Extend and develop

Runners and judges implement small async contracts and register through public factories.
See [custom_runner.py](examples/custom_runner.py) for an offline example and
[architecture](docs/architecture.md) for module boundaries.

```bash
python examples/custom_runner.py
python -m pytest
python -m ruff check src tests scripts examples
python -m build
```

Install only the extras you need: `adaptive` for training, `analysis` for graph diagnostics,
`data` for dataset downloads, and `dev` for development tools.
GateKG’s original code uses the [MIT license](LICENSE). Adapted evaluation code and prompts have [third-party notices](THIRD_PARTY_NOTICES.md).
