# Configuration

Configuration is selected from an explicit `--config` path, `GATEKG_CONFIG`, a local
`config.yaml`, `configs/default.yaml`, or the packaged default, in that order.
Environment variables override the selected YAML. Secrets belong in `.env` or the process environment.

## Model presets

The local preset uses [Qwen 3.5 on Ollama](https://ollama.com/library/qwen3.5).
The vLLM preset uses [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
The Google preset uses the stable [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).
These are configurable starting points, not hardware or quality guarantees.

Local context limits are deliberately conservative. Raise them only when your serving
process and available memory support the larger context. Use a current Ollama or vLLM
release that supports the selected model.

`configs/google.yaml` and `configs/vllm.yaml` show minimal provider configurations.
Omitted provider settings use the defaults defined by the configuration loader.
The embedding models stay explicitly configured to preserve vector dimensions.
Changing an embedding model or dimension requires a separate graph/cache or a rebuild.

## Overrides

- `GATEKG_PROVIDER`: `google`, `ollama`, or `vllm`.
- `GATEKG_LLM_MODEL`: model for inference.
- `GATEKG_KG_MODEL` and `GATEKG_KG_PROVIDER`: graph extraction model.
- `GATEKG_JUDGE_MODEL`: Google evaluation model.
- `GATEKG_EMBEDDING_PROVIDER`: `google` or `ollama`.
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`: Gemini API access.
- `VERTEX_API_KEY`: Vertex access when `api.google.provider` is `vertex_ai`.
- `OLLAMA_BASE_URL` and `VLLM_BASE_URL`: local service endpoints.
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: graph storage.

Use `GATEKG_HOME` for the runtime root. The `GATEKG_DATA_DIR`,
`GATEKG_ARTIFACT_DIR`, `GATEKG_LOG_DIR`, `GATEKG_CACHE_DIR`, and
`GATEKG_OUTPUT_DIR` variables override individual runtime directories.
`GATEKG_POLICY_ARTIFACT_DIR` selects the trained policy bundle.

Batch dataset and output paths are relative to the manifest directory. Registered
dataset names still resolve through the runtime data directory. CLI paths are relative
to the current directory.

## vLLM

Install vLLM separately in a GPU-compatible serving environment, alongside GateKG:

```bash
python -m gatekg.serving --config configs/vllm.yaml --print-command
python -m gatekg.serving --config configs/vllm.yaml
```

Additional arguments are forwarded to `vllm serve`. The server binds to localhost by
default. Configure authentication before exposing it to another machine.
