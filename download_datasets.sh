#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN_LITERARYQA="${PYTHON_BIN_LITERARYQA:-$PYTHON_BIN}"
CACHE_DIR="${CACHE_DIR:-data/hf_cache}"
OUTPUT_BASE="${OUTPUT_BASE:-data/processed}"
FORCE="${FORCE:-0}"
DOCS_DOWNSAMPLE="${DOCS_DOWNSAMPLE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      ;;
    --docs-downsample)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        printf 'Usage: %s [--force] [--docs-downsample N]\n' "$(basename "$0")" >&2
        exit 1
      fi
      DOCS_DOWNSAMPLE="$2"
      shift
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      printf 'Usage: %s [--force] [--docs-downsample N]\n' "$(basename "$0")" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "$CACHE_DIR" "$OUTPUT_BASE"

if [[ -n "$DOCS_DOWNSAMPLE" ]]; then
  if ! [[ "$DOCS_DOWNSAMPLE" =~ ^[1-9][0-9]*$ ]]; then
    printf 'DOCS_DOWNSAMPLE / --docs-downsample must be a positive integer (got: %s)\n' "$DOCS_DOWNSAMPLE" >&2
    exit 1
  fi
fi

log() {
  printf '[download_datasets] %s\n' "$*"
}

run_prepare() {
  local dataset="$1"
  local split="$2"
  local loader_kwargs="${3:-}"
  local python_bin="${4:-$PYTHON_BIN}"
  local docs_downsample="${5:-$DOCS_DOWNSAMPLE}"
  local output_split="${6:-$split}"
  local output_dir="$OUTPUT_BASE/$dataset/$output_split"
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

  if [[ -n "$docs_downsample" ]]; then
    cmd+=(--docs-downsample "$docs_downsample")
  fi

  if [[ "$FORCE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  log "Preparing '$dataset' ($split) with $python_bin -> $output_dir (docs_downsample=${docs_downsample:-<none>})"
  "${cmd[@]}"
}

run_merge() {
  local dataset="$1"
  local split="$2"
  local python_bin="${3:-$PYTHON_BIN}"
  local input_dir="$OUTPUT_BASE/$dataset/$split"
  local merged_dir="$OUTPUT_BASE/$dataset/${split}_merged"
  local merged_docs_path="$merged_dir/documents/documents.jsonl"
  local merged_queries_path="$merged_dir/queries/queries.jsonl"

  if [[ ! -f "$input_dir/documents/documents.jsonl" || ! -f "$input_dir/queries/queries.jsonl" ]]; then
    log "Skipping merge for '$dataset' ($split): source files missing in $input_dir"
    return 0
  fi

  if [[ "$FORCE" != "1" && -f "$merged_docs_path" && -f "$merged_queries_path" ]]; then
    log "Skipping merge for '$dataset' ($split): outputs already exist at $merged_dir"
    return 0
  fi

  local cmd=(
    "$python_bin" src/document_merger/merge_dataset.py
    --dataset-path "$input_dir"
  )
  if [[ "$FORCE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  log "Merging '$dataset' ($split) -> ${split}_merged"
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
log "DOCS_DOWNSAMPLE=${DOCS_DOWNSAMPLE:-<none>}"
log "Datasets: natural_questions/validation_300_docs, natural_questions/validation_100_docs, novelqa/public, poquad/train, poquad/validation, squad/train(docs=100), squad/validation, triviaqa_span_annotated/train_artificial_1000_docs, triviaqa_span_annotated/test_artificial_1000_docs, gutenqa/all, literaryqa/validation, literaryqa/test, qasper/train(docs=400), qasper/validation, qasper/test"
log "Merging after download: poquad/train -> train_merged, poquad/validation -> validation_merged, gutenqa/all -> all_merged, triviaqa_span_annotated/train_artificial_1000_docs -> train_artificial_1000_docs_merged, triviaqa_span_annotated/test_artificial_1000_docs -> test_artificial_1000_docs_merged"
log_literaryqa_requirements

run_prepare "natural_questions" "validation" "" "$PYTHON_BIN" "300" "validation_300_docs"
run_prepare "natural_questions" "validation" "" "$PYTHON_BIN" "100" "validation_100_docs"
run_prepare "novelqa" "public" "{download_if_missing: false}"
run_prepare "poquad" "train"
run_prepare "poquad" "validation"
run_prepare "squad" "train" "" "$PYTHON_BIN" "100"
run_prepare "squad" "validation"
run_prepare "triviaqa_span_annotated" "train_artificial" "" "$PYTHON_BIN" "1000" "train_artificial_1000_docs"
run_prepare "triviaqa_span_annotated" "test_artificial" "" "$PYTHON_BIN" "1000" "test_artificial_1000_docs"
run_prepare "gutenqa" "all"
run_prepare "literaryqa" "validation" "" "$PYTHON_BIN_LITERARYQA"
run_prepare "literaryqa" "test" "" "$PYTHON_BIN_LITERARYQA"
run_prepare "qasper" "train" "" "$PYTHON_BIN" "400"
run_prepare "qasper" "validation"
run_prepare "qasper" "test"

run_merge "poquad" "train"
run_merge "poquad" "validation"
run_merge "gutenqa" "all"
run_merge "triviaqa_span_annotated" "train_artificial_1000_docs"
run_merge "triviaqa_span_annotated" "test_artificial_1000_docs"

log "Dataset download run complete."
