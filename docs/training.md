# Policy training

Collect comparable runs for the same questions with `read_only_rlm` and `read_write_rlm`. The dataset builder combines traces, evaluation results, and graph-health snapshots.

For three-mode experiments, also collect `aggressive_write_rlm` runs and pass `--include-aggressive-write` to training. Existing three-mode policies retain their saved label maps.

Keep each log beside its `.manifest.json` input snapshot and its graph-tool trace
(`.trajectory.jsonl`). Preserve the health
database under `GATEKG_CACHE_DIR`. Missing health snapshots are reported as partial
training rows. Runs without input manifests are not included.

```bash
gatekg-build-dataset --log-root logs --output data/gatekg/runs.jsonl
gatekg-train --dataset data/gatekg/runs.jsonl --artifact-dir artifacts/gatekg/shared --group-split
gatekg-interpret dataset --dataset data/gatekg/runs.jsonl
```

The selector learns which graph policy to use. The tool gate learns whether to allow
a proposed graph operation. Reward-based labels compare matched baseline trajectories.
Trajectory labels instead use a Google judge:

```bash
gatekg-label --dataset data/gatekg/runs.jsonl
gatekg-train --dataset data/gatekg/runs.labeled.jsonl --gate-label-source trajectory_state --group-split
```

Graph runs write one private trace record per question. Each executed call carries
the positional and keyword values passed to the graph client and either its return
value or the exception type. Console tool summaries show counts, keyword names and
result types. Capture limits each argument collection and each result to 8,192
string characters, 4,096 per string, 32 entries per container and eight nesting
levels. Omitted values carry explicit markers. Recognized credential fields such
as `api_key`, `password` and `access_token` are redacted by key. Free-text queries,
memory bodies and graph results can still contain sensitive data. Trace files,
training JSONL and the trajectory-label cache are written with owner-only permissions.
Running `gatekg-label` sends the captured call evidence to the configured Google judge.

For policy runs, both label sources use the gate counters recorded at proposal time.
A blocked write contributes to the history of the next executed call. Only executed
calls receive labels. For older runs the builder pairs console tool calls with
recorded allow decisions and retains the original counters. Rebuild older policy
datasets from their logs to use this alignment. For these older calls the judge sees
the available console summaries.

For a complete training round:

```bash
gatekg-advance --log-root logs --round 1
```

Add `--gate-label-source trajectory_state` to enable trajectory labeling.
Each round writes dataset snapshots, training metrics, model weights, feature metadata,
label maps, and a round manifest. Set `--artifact-dir` explicitly when selecting a bundle.

Training requires useful variation across runs. Inspect dataset summaries and validation
metrics before deploying a policy. The default split groups normalized question text across calls, models and dataset variants. Both models use the same question partition. With at least five distinct questions, 20% of the groups (rounded down) are held out. Smaller runs train on all questions and have no validation estimate. Each training run writes the identities it used to `split.json` alongside the models.

This partition measures performance on held-out questions. For a claim about generalization to unseen documents, partition source documents before collecting training and evaluation runs, including documents used for padding. Different questions about a shared document can land in different question groups.
