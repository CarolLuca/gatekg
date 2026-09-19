"""Launch an external vLLM installation using GateKG configuration."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
from collections.abc import Sequence

from gatekg.utils.config import load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--print-command", action="store_true")
    args, extra = parser.parse_known_args(argv)
    config = load_config(args.config).vllm
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    command = [
        "vllm",
        "serve",
        config.llm_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
    ]
    if config.quantization:
        command.extend(["--quantization", config.quantization])
    if config.language_model_only:
        command.append("--language-model-only")
    command.extend(extra)
    if args.print_command:
        print(shlex.join(command))
        return 0
    executable = shutil.which("vllm")
    if executable is None:
        parser.error("Install vLLM in the serving environment before starting the server.")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": config.cuda_visible_devices,
    }
    if config.api_key:
        environment["VLLM_API_KEY"] = config.api_key
    os.execvpe(executable, command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
