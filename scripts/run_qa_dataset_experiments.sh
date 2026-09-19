#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?Usage: $0 <dataset|all> [entry_limit]}"
ENTRY_LIMIT="${2:-0}"
PYTHON_BIN="${PYTHON:-python3}"
RUNNERS_CSV="${GATEKG_RUNNERS:-llm,base_rlm,custom_base_rlm,read_only_rlm,read_write_rlm,aggressive_write_rlm,gatekg}"

export GATEKG_HOME="${GATEKG_HOME:-${ROOT_DIR}}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATASETS=(two_wiki_multihopqa hotpotqa squad novelhopqa mquake)
if [[ "${DATASET}" != "all" ]]; then
  DATASETS=("${DATASET}")
fi
IFS=',' read -r -a RUNNERS <<< "${RUNNERS_CSV}"

for dataset_name in "${DATASETS[@]}"; do
  dataset_file="${GATEKG_DATA_DIR:-${GATEKG_HOME}/data}/${dataset_name}/qa.jsonl"
  if [[ ! -f "${dataset_file}" ]]; then
    download_args=(--datasets "${dataset_name}")
    if [[ "${ENTRY_LIMIT}" -gt 0 ]]; then
      download_args+=(--limit "${ENTRY_LIMIT}")
    fi
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_qa_datasets.py" "${download_args[@]}"
  fi

  for runner in "${RUNNERS[@]}"; do
    args=(run "${dataset_file}" "${runner}" --format qa_jsonl --limit "${ENTRY_LIMIT}")
    if [[ "${runner}" == "gatekg" && -n "${GATEKG_POLICY_ARTIFACT_DIR:-}" ]]; then
      args+=(--artifact-dir "${GATEKG_POLICY_ARTIFACT_DIR}")
    fi
    "${PYTHON_BIN}" -m gatekg "${args[@]}"
  done
done
