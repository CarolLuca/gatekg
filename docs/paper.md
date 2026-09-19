# Running the paper workflow

The paper profile uses Gemini 3 Flash for inference and Gemini 3.1 Pro for answer and trajectory judgments. Its model identifiers select preview endpoints and require access from your Google project. The general-purpose defaults are in `configs/default.yaml`.

## Environment and data

Use CPython 3.11.14 on x86-64 Linux for the pinned CPU environment. This environment supports graph clients, policy training and the analysis scripts. Serve local language models in their own environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/paper-linux-cpu.txt
python -m pip install --no-build-isolation --no-deps -e .
cp .env.example .env
export GATEKG_CONFIG="$PWD/configs/paper.yaml"
```

Set the Google and Neo4j credentials in `.env` and start the graph database with `docker compose up -d neo4j`. The following commands acquire the 1,101 LooGLE long-dependency questions and generate context-length variants. The pinned upstream JSONL already contains `id`, `doc_id`, `context`, `question` and `answer`. It includes questions with answer choices. The workflow preserves the upstream wording and answer encoding. The builder writes up to eleven large JSONL files and reports their actual average estimated lengths. These estimates use word counts.

```bash
mkdir -p data/loogle
curl --fail --location --retry 3 --output data/loogle/longdep_qa.download \
  https://huggingface.co/datasets/bigai-nlco/LooGLE/resolve/4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e/data/longdep_qa.jsonl
echo '6c047bc6e904e5fd632e44a2f2624d75e85fe0f871c6b74bcfffa918bf2f3624  data/loogle/longdep_qa.download' | sha256sum --check && \
  mv data/loogle/longdep_qa.download data/loogle/longdep_qa.jsonl
python scripts/build_loogle_extended_datasets.py
```

The transfer and LooGLE v2 download scripts use pinned source revisions. Consult the upstream dataset terms before sharing derived data. [Original LooGLE](https://huggingface.co/datasets/bigai-nlco/LooGLE) · [LooGLE v2](https://huggingface.co/datasets/MuLabPKU/LooGLE-v2)

## Routing and training

The selector has two outputs. With the seven standard health measures, its layer widths are 74, 128, 64 and 2. The gate uses widths 110, 128, 64 and 1. The gate threshold is 0.5. Both models reuse the baseline health vector throughout a question, while the gate updates tool-attempt counters after each proposal. Whole-database community and schema maintenance are administrative operations.

Use a dedicated log directory for policy training. Collect read-only and read-write trajectories, train the first policy, then collect trajectories under that policy before advancing. The third round applies step-level labels and backward filling to the combined pool. The round command sets selector dropout to 0.2, gate dropout to 0.1, AdamW weight decay to 0.01, and early-stopping patience to five epochs. Patience, optimizer learning rate (0.001), epoch caps (50 and 30), and training seed (0) are explicit implementation settings.

```bash
export GATEKG_LOG_DIR="$PWD/logs/train"
qa=data/loogle/longdep_qa.jsonl
gatekg run "$qa" read_only_rlm
gatekg run "$qa" read_write_rlm
gatekg-advance --round 1 --log-root "$GATEKG_LOG_DIR"
rounds=artifacts/gatekg/rounds
gatekg run "$qa" gatekg --artifact-dir "$rounds/round_001"
gatekg-advance --round 2 --log-root "$GATEKG_LOG_DIR"
gatekg-advance --round 3 --log-root "$GATEKG_LOG_DIR" --gate-label-source trajectory_state
export GATEKG_LOG_DIR="$PWD/logs/evaluation"
```

## Table workflows

| Paper analysis | Command or inputs |
| --- | --- |
| Context scaling and write sensitivity | Run `gatekg run data/loogle/longdep_qa_extended_50k.jsonl read_write_rlm --context-field extended_context --kg-context-field initial_context`. Repeat over target lengths and runners. Choose the same question IDs for paired comparisons. |
| LooGLE v2 | Set `GATEKG_RUNNERS=read_only_rlm,read_write_rlm,gatekg` and `GATEKG_POLICY_ARTIFACT_DIR` to the round being evaluated. Export both variables and run `bash scripts/run_loogle_v2_experiments.sh`. |
| Cross-dataset QA | With DeepSeek-R1 14B running locally, export `GATEKG_PROVIDER=ollama`, `GATEKG_LLM_MODEL=deepseek-r1:14b` and `GATEKG_KG_PROVIDER=google`. Run `bash scripts/run_qa_dataset_experiments.sh hotpotqa 100`. Repeat for `mquake`, `two_wiki_multihopqa` and `novelhopqa`. The example evaluates 100 entries. |
| Training rounds | Read the selector and gate train/validation metrics emitted by each `gatekg-advance` invocation. |
| Bootstrap intervals | Run `python scripts/analyze_evaluation.py bootstrap RUN.json` on a completed run report. It resamples questions 10,000 times with seed 42 and reports percentile intervals for mean judge score and token F1 on the 0–1 scale. Each interval describes one system. |
| Judge agreement | Run `python scripts/analyze_evaluation.py audit ratings.csv` with one paired binary decision per audited output in columns `judge,human`. The script reports agreement and Cohen’s kappa. |
| Feature ablations and classifier comparisons | Run `python scripts/compare_gates.py "$rounds/round_003/dataset.jsonl" "$rounds/round_003" --semantic`. The script checks the partition, zeroes feature groups at inference, and reports accuracy and ROC AUC on held-out calls. |

For Qwen runs, select the Ollama provider and the intended Qwen model with the same environment overrides. Configure its context window and input cap for the selected context lengths. For the classifier comparison, the script freezes the trained gate’s feature embeddings, fits logistic regression, random forest and a two-hidden-layer MLP on the training partition, and optionally replaces the question hash with the pinned 384-dimensional MiniLM encoder. It reports the fitted GateKG head separately. The fixed step/duplicate rule requires the raw duplicate ratio for each validation call and uses one-based steps. These scripts compute statistics from the runs you supply.

The fitted-gate ablations operate on the input to the first classifier layer. Each
call passes through the trained embeddings and numeric LayerNorm once. Starting
from that same vector, `zero_health` masks average health and all health metrics,
`zero_trajectory` masks the six call-history and write-mode coordinates, and
`zero_question` masks the question embedding. Each mask sets just its coordinates
to zero and copies every other coordinate from the full input. The same fitted
head scores all four inputs.
