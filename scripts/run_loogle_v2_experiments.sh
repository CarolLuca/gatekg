#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENTRY_LIMIT="${1:-0}"
PYTHON_BIN="${PYTHON:-python3}"
RUNNERS_CSV="${GATEKG_RUNNERS:-llm,base_rlm,custom_base_rlm,read_only_rlm,read_write_rlm,aggressive_write_rlm,gatekg}"

export GATEKG_HOME="${GATEKG_HOME:-${ROOT_DIR}}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

dataset_file="${GATEKG_DATA_DIR:-${GATEKG_HOME}/data}/loogle_v2/qa.jsonl"
if [[ ! -f "${dataset_file}" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_loogle_v2.py"
fi

IFS=',' read -r -a RUNNERS <<< "${RUNNERS_CSV}"
for runner in "${RUNNERS[@]}"; do
  args=(run "${dataset_file}" "${runner}" --format qa_jsonl --limit "${ENTRY_LIMIT}")
  if [[ "${runner}" == "gatekg" && -n "${GATEKG_POLICY_ARTIFACT_DIR:-}" ]]; then
    args+=(--artifact-dir "${GATEKG_POLICY_ARTIFACT_DIR}")
  fi
  "${PYTHON_BIN}" -m gatekg "${args[@]}"
done

