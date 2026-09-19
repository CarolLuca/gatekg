"""Centralized, environment-aware runtime paths.

GateKG never assumes that datasets, logs, caches, or learned artifacts live
inside an installed Python package. Applications can place each directory on
separate storage by setting the corresponding ``GATEKG_*`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_from(base: Path, value: str | None, default: str) -> Path:
    candidate = Path(value or default).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def discover_project_root(start: str | Path | None = None) -> Path:
    """Return the runtime project root without depending on package layout."""

    configured = os.getenv("GATEKG_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    origin = Path(start or Path.cwd()).expanduser().resolve()
    if origin.is_file():
        origin = origin.parent
    for candidate in (origin, *origin.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "gatekg").is_dir():
            return candidate
    return origin


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    root: Path
    data: Path
    artifacts: Path
    logs: Path
    cache: Path
    outputs: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> RuntimePaths:
        project_root = discover_project_root(root)
        return cls(
            root=project_root,
            data=_resolve_from(project_root, os.getenv("GATEKG_DATA_DIR"), "data"),
            artifacts=_resolve_from(project_root, os.getenv("GATEKG_ARTIFACT_DIR"), "artifacts"),
            logs=_resolve_from(project_root, os.getenv("GATEKG_LOG_DIR"), "logs"),
            cache=_resolve_from(project_root, os.getenv("GATEKG_CACHE_DIR"), ".cache"),
            outputs=_resolve_from(project_root, os.getenv("GATEKG_OUTPUT_DIR"), "outputs"),
        )

    def create_runtime_dirs(self) -> None:
        """Create writable output directories; input data is intentionally omitted."""

        for path in (self.artifacts, self.logs, self.cache, self.outputs):
            path.mkdir(parents=True, exist_ok=True)


def runtime_paths(root: str | Path | None = None) -> RuntimePaths:
    return RuntimePaths.from_env(root)


def policy_artifact_dir(value: str | Path | None = None) -> Path:
    """Resolve a policy bundle consistently across training and inference."""
    paths = runtime_paths()
    configured = str(value) if value is not None else os.getenv("GATEKG_POLICY_ARTIFACT_DIR")
    if configured:
        return _resolve_from(paths.root, configured, "")
    return paths.artifacts / "gatekg" / "shared"
