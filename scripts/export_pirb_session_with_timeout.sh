#!/usr/bin/env bash

set -u -o pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_pirb_session_with_timeout.sh \
    --session-path <path> \
    [--timeout-seconds 300] \
    [--python-bin .venv/bin/python] \
    [--output-root export_to_pirb] \
    [--report-root export_to_pirb/_batch_reports] \
    [--run-name run_0002] [--run-name run_0011]

Description:
  Runs PIRB export separately for each run_* directory directly under
  --session-path. Each run is capped by timeout; timed-out runs are skipped.
  Generates TSV + text reports with finished/skipped/error statuses.
  When --run-name is provided one or more times, only selected run_* folders
  are processed.
EOF
}

SESSION_PATH=""
TIMEOUT_SECONDS=300
PYTHON_BIN=".venv/bin/python"
OUTPUT_ROOT="export_to_pirb"
REPORT_ROOT="export_to_pirb/_batch_reports"
RUN_NAMES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-path)
      SESSION_PATH="${2:-}"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --report-root)
      REPORT_ROOT="${2:-}"
      shift 2
      ;;
    --run-name)
      RUN_NAMES+=("${2:-}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$SESSION_PATH" ]]; then
  echo "--session-path is required" >&2
  usage
  exit 2
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "Missing required command: timeout" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -d "$SESSION_PATH" ]]; then
  echo "Session path does not exist: $SESSION_PATH" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 2

mapfile -t RUN_DIRS < <(find "$SESSION_PATH" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | sort)
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "No run_* subfolders found under: $SESSION_PATH" >&2
  exit 2
fi

if [[ "${#RUN_NAMES[@]}" -gt 0 ]]; then
  declare -A RUN_DIR_BY_NAME=()
  for run_dir in "${RUN_DIRS[@]}"; do
    run_name="$(basename "$run_dir")"
    RUN_DIR_BY_NAME["$run_name"]="$run_dir"
  done

  FILTERED_RUN_DIRS=()
  MISSING_RUN_NAMES=()
  declare -A ADDED_RUN_NAMES=()
  for requested_name in "${RUN_NAMES[@]}"; do
    candidate_dir="${RUN_DIR_BY_NAME[$requested_name]:-}"
    if [[ -z "$candidate_dir" ]]; then
      MISSING_RUN_NAMES+=("$requested_name")
      continue
    fi
    if [[ -n "${ADDED_RUN_NAMES[$requested_name]:-}" ]]; then
      continue
    fi
    ADDED_RUN_NAMES["$requested_name"]=1
    FILTERED_RUN_DIRS+=("$candidate_dir")
  done

  if [[ "${#MISSING_RUN_NAMES[@]}" -gt 0 ]]; then
    printf 'Warning: requested --run-name not found under session: %s\n' \
      "${MISSING_RUN_NAMES[*]}" >&2
  fi

  if [[ "${#FILTERED_RUN_DIRS[@]}" -eq 0 ]]; then
    echo "No valid selected runs found under: $SESSION_PATH" >&2
    exit 2
  fi

  RUN_DIRS=("${FILTERED_RUN_DIRS[@]}")
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$REPORT_ROOT/logs_$TS"
REPORT_TSV="$REPORT_ROOT/pirb_per_run_$TS.tsv"
SUMMARY_TXT="$REPORT_ROOT/summary_$TS.txt"
SKIPPED_TXT="$REPORT_ROOT/skipped_$TS.txt"

mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_ROOT"

echo -e "run\tstatus\texit_code\tduration_sec\tsuccessfully_converted\tfailed\tnote\tlog_path" > "$REPORT_TSV"

total_runs="${#RUN_DIRS[@]}"
echo "Starting session export batch"
echo "session_path=$SESSION_PATH"
echo "runs_total=$total_runs"
echo "timeout_seconds=$TIMEOUT_SECONDS"
echo "python_bin=$PYTHON_BIN"
echo "output_root=$OUTPUT_ROOT"
echo "report_tsv=$REPORT_TSV"
echo "log_dir=$LOG_DIR"
echo

idx=0
for run_dir in "${RUN_DIRS[@]}"; do
  idx=$((idx + 1))
  run_name="$(basename "$run_dir")"
  log_path="$LOG_DIR/${run_name}.log"

  start_epoch="$(date +%s)"
  timeout "${TIMEOUT_SECONDS}s" "$PYTHON_BIN" run_annotate_and_convert.py \
    --input-path "$run_dir" \
    --output-root "$OUTPUT_ROOT" \
    --overwrite-run-dir \
    >"$log_path" 2>&1
  rc=$?
  end_epoch="$(date +%s)"
  duration_sec=$((end_epoch - start_epoch))

  succ="$(grep -E "^- successfully converted:" "$log_path" | tail -n1 | awk -F': ' '{print $2}' || true)"
  fail="$(grep -E "^- failed:" "$log_path" | tail -n1 | awk -F': ' '{print $2}' || true)"
  [[ -z "$succ" ]] && succ="NA"
  [[ -z "$fail" ]] && fail="NA"

  status=""
  note=""
  if [[ "$rc" -eq 124 ]]; then
    status="timed_out"
    note=">${TIMEOUT_SECONDS}s"
  elif [[ "$rc" -eq 0 ]]; then
    if [[ "$succ" == "1" && "$fail" == "0" ]]; then
      status="finished"
    elif [[ "$fail" != "0" && "$fail" != "NA" ]]; then
      status="skipped_invalid"
      note="$(grep -E '^  reason:' "$log_path" | head -n1 | sed 's/^  reason: //' || true)"
    else
      status="finished_with_warnings"
      note="check_log"
    fi
  else
    status="error"
    note="exit=$rc"
  fi

  note="${note//$'\t'/ }"
  echo -e "${run_name}\t${status}\t${rc}\t${duration_sec}\t${succ}\t${fail}\t${note}\t${log_path}" >> "$REPORT_TSV"
  echo "[$idx/$total_runs] $run_name | $status | rc=$rc | ${duration_sec}s"
done

finished_count="$(awk -F'\t' 'NR>1 && $2=="finished" {c++} END{print c+0}' "$REPORT_TSV")"
timed_out_count="$(awk -F'\t' 'NR>1 && $2=="timed_out" {c++} END{print c+0}' "$REPORT_TSV")"
skipped_invalid_count="$(awk -F'\t' 'NR>1 && $2=="skipped_invalid" {c++} END{print c+0}' "$REPORT_TSV")"
finished_warn_count="$(awk -F'\t' 'NR>1 && $2=="finished_with_warnings" {c++} END{print c+0}' "$REPORT_TSV")"
error_count="$(awk -F'\t' 'NR>1 && $2=="error" {c++} END{print c+0}' "$REPORT_TSV")"

{
  echo "session_path=$SESSION_PATH"
  echo "runs_total=$total_runs"
  echo "finished=$finished_count"
  echo "timed_out=$timed_out_count"
  echo "skipped_invalid=$skipped_invalid_count"
  echo "finished_with_warnings=$finished_warn_count"
  echo "error=$error_count"
  echo "report_tsv=$REPORT_TSV"
  echo "log_dir=$LOG_DIR"
} | tee "$SUMMARY_TXT"

{
  echo "Skipped runs (timed_out + skipped_invalid):"
  awk -F'\t' 'NR>1 && ($2=="timed_out" || $2=="skipped_invalid") {print "- " $1 " (" $2 "): " $7}' "$REPORT_TSV"
} | tee "$SKIPPED_TXT"

echo "summary_txt=$SUMMARY_TXT"
echo "skipped_txt=$SKIPPED_TXT"
