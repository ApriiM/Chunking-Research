#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"

# Default semantic chunker model(s).
SENTENCE_MODELS="${SENTENCE_MODELS:-BAAI/bge-m3}"

# Opt-in because the default LumberChunker model is much larger.
DOWNLOAD_LUMBER_MODEL="${DOWNLOAD_LUMBER_MODEL:-0}"
LUMBER_MODEL="${LUMBER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"

log() {
  printf '[download_models] %s\n' "$*"
}

log "Using python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
log "HF cache: $HF_HOME"

if [[ -n "$HUGGINGFACE_HUB_TOKEN" ]]; then
  export HUGGINGFACE_HUB_TOKEN
  log "HUGGINGFACE_HUB_TOKEN is set"
else
  log "HUGGINGFACE_HUB_TOKEN is not set"
fi

export HF_HOME
export SENTENCE_MODELS
export LUMBER_MODEL

log "Downloading sentence embedding model(s): $SENTENCE_MODELS"
"$PYTHON_BIN" - <<'PY'
import os
from sentence_transformers import SentenceTransformer

models = [m for m in os.environ["SENTENCE_MODELS"].split(",") if m.strip()]
for model_name in models:
    model_name = model_name.strip()
    print(f"[download_models] downloading sentence model: {model_name}")
    SentenceTransformer(model_name)
    print(f"[download_models] ready: {model_name}")
PY

if [[ "$DOWNLOAD_LUMBER_MODEL" == "1" ]]; then
  log "Downloading LumberChunker causal LM: $LUMBER_MODEL"
  "$PYTHON_BIN" - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = os.environ["LUMBER_MODEL"].strip()
token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

print(f"[download_models] downloading tokenizer: {model_name}")
AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)

print(f"[download_models] downloading causal LM: {model_name}")
AutoModelForCausalLM.from_pretrained(model_name, token=token, trust_remote_code=True)

print(f"[download_models] ready: {model_name}")
PY
else
  log "Skipping LumberChunker model download (set DOWNLOAD_LUMBER_MODEL=1 to enable)"
fi

log "Model download run complete."
