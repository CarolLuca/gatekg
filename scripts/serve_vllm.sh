#!/usr/bin/env bash
set -euo pipefail

exec "${PYTHON_BIN:-python}" -m gatekg.serving "$@"
