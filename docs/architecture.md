# Architecture

- `datasets` loads validated QA records and handles field mapping and chunking.
- `contracts`, `registry`, and `judges` define extension interfaces.
- `experiments` owns runner lifecycle, evaluation, and result persistence.
- `records` stores portable run inputs.
- `api` implements model and graph integrations, with a shared connection lifecycle for direct and tool-based graph access.
- `training` builds datasets, labels traces, trains policies, and interprets results.
- `health` measures and stores graph diagnostics.
- `paths` and `utils.config` centralize runtime configuration.

A runner implements `start`, `predict`, and `close`. Register a `RunnerDefinition`
with `register_runner`; its factory receives a `RunnerContext`.
A judge implements `ajudge` and returns a binary `JudgeResult`.
Use `register_judge` to add a judge factory.

Factories are called for each experiment. Runner cleanup runs even when prediction
fails. Provider retries are bounded, and exhausted calls fail the run instead of becoming predictions. Experiment execution is synchronous at the public orchestration boundary;
run separate experiments in separate processes. Configure the environment before
constructing provider clients.

Dataset downloads and graph-health reporting remain standalone scripts. Core training
commands ship in the wheel and do not require a source checkout.
