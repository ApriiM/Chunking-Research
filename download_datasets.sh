#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN_LITERARYQA="${PYTHON_BIN_LITERARYQA:-$PYTHON_BIN}"
CACHE_DIR="${CACHE_DIR:-data/hf_cache}"
OUTPUT_BASE="${OUTPUT_BASE:-data/processed}"
FORCE="${FORCE:-0}"

for arg in "$@"; do
  case "$arg" in
    --force)
      FORCE=1
      ;;
    *)
      printf 'Unknown argument: %s\n' "$arg" >&2
      printf 'Usage: %s [--force]\n' "$(basename "$0")" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$CACHE_DIR" "$OUTPUT_BASE"

log() {
  printf '[download_datasets] %s\n' "$*"
}

run_prepare() {
  local dataset="$1"
  local split="$2"
  local loader_kwargs="${3:-}"
  local python_bin="${4:-$PYTHON_BIN}"
  local output_dir="$OUTPUT_BASE/$dataset/$split"
  local docs_path="$output_dir/documents/documents.jsonl"
  local queries_path="$output_dir/queries/queries.jsonl"

  if [[ "$FORCE" != "1" && -f "$docs_path" && -f "$queries_path" ]]; then
    log "Skipping '$dataset' ($split): outputs already exist at $output_dir"
    return 0
  fi

  local cmd=(
    "$python_bin" -m src.data_loader.prepare_dataset
    --dataset "$dataset"
    --split "$split"
    --cache-dir "$CACHE_DIR"
    --output-dir "$output_dir"
  )

  if [[ -n "$loader_kwargs" ]]; then
    cmd+=(--loader-kwargs "$loader_kwargs")
  fi

  if [[ "$FORCE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  log "Preparing '$dataset' ($split) with $python_bin -> $output_dir"
  "${cmd[@]}"
}

log_literaryqa_requirements() {
  log "LiteraryQA requires a dedicated Python >= 3.12 environment."
  log "Recommended install: .venv-literaryqa/bin/pip install -r requirements-literaryqa.txt"
  log "Set PYTHON_BIN_LITERARYQA to that venv's python if it differs from PYTHON_BIN."
}

log "Using python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
log "Using python for LiteraryQA: $("$PYTHON_BIN_LITERARYQA" -c 'import sys; print(sys.executable)')"
log "Cache dir: $CACHE_DIR"
log "Output base: $OUTPUT_BASE"
log "FORCE=$FORCE"
log "Datasets: natural_questions/validation, novelqa/public, novelqa/copyright, poquad/train, poquad/validation, squad/train, squad/validation, triviaqa_span_annotated/train_artificial, triviaqa_span_annotated/test_artificial, gutenqa/all, gutenqa_concat/all, literaryqa/train, literaryqa/validation, literaryqa/test, qasper/train, qasper/validation, qasper/test"
log_literaryqa_requirements

run_prepare "natural_questions" "validation"
run_prepare "novelqa" "public" "{download_if_missing: false}"
run_prepare "novelqa" "copyright" "{download_if_missing: false}"
run_prepare "poquad" "train"
run_prepare "poquad" "validation"
run_prepare "squad" "train"
run_prepare "squad" "validation"
run_prepare "triviaqa_span_annotated" "train_artificial"
run_prepare "triviaqa_span_annotated" "test_artificial"
run_prepare "gutenqa" "all"
run_prepare "gutenqa_concat" "all"
run_prepare "literaryqa" "train" "" "$PYTHON_BIN_LITERARYQA"
run_prepare "literaryqa" "validation" "" "$PYTHON_BIN_LITERARYQA"
run_prepare "literaryqa" "test" "" "$PYTHON_BIN_LITERARYQA"
run_prepare "qasper" "train"
run_prepare "qasper" "validation"
run_prepare "qasper" "test"

log "Dataset download run complete."
