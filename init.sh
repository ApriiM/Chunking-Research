#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MAIN_VENV=".venv"
LITERARY_VENV=".venv-literaryqa"

MAIN_PY_VERSION="3.10.18"
LITERARY_PY_VERSION="3.12.11"

MAIN_FREEZE="versions_venv.txt"
LITERARY_FREEZE="versions_venv_literaryqa.txt"

DOWNLOAD_NOVELQA=0
NOVELQA_FORCE=0
NOVELQA_HF_TOKEN=""
NOVELQA_REPO="NovelQA/NovelQA"
NOVELQA_TARGET="downloads/NovelQA"

usage() {
  cat <<'EOF'
Usage: ./init.sh [options]

Options:
  --download-novelqa        Clone NovelQA as the last setup step.
  --hf-token TOKEN          Hugging Face token for NovelQA clone.
                            Fallbacks: HUGGINGFACE_HUB_TOKEN, HF_TOKEN.
  --force-novelqa           Re-download NovelQA even if target exists.
  --novelqa-repo REPO       HF dataset repo slug (default: NovelQA/NovelQA).
  --novelqa-target PATH     Clone destination (default: downloads/NovelQA).
  -h, --help                Show this help.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --download-novelqa)
        DOWNLOAD_NOVELQA=1
        ;;
      --force-novelqa)
        NOVELQA_FORCE=1
        ;;
      --hf-token)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --hf-token" >&2
          exit 1
        fi
        NOVELQA_HF_TOKEN="$2"
        shift
        ;;
      --novelqa-repo)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --novelqa-repo" >&2
          exit 1
        fi
        NOVELQA_REPO="$2"
        shift
        ;;
      --novelqa-target)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --novelqa-target" >&2
          exit 1
        fi
        NOVELQA_TARGET="$2"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
    shift
  done
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

ensure_pyenv_python() {
  local version="$1"
  local pyenv_root="$2"
  if ! pyenv versions --bare | grep -Fxq "$version"; then
    echo "Installing Python $version with pyenv" >&2
    pyenv install -s "$version" >&2
  fi

  local py_bin="$pyenv_root/versions/$version/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "pyenv python not found for $version: $py_bin" >&2
    exit 1
  fi
  printf '%s\n' "$py_bin"
}

ensure_venv() {
  local venv_dir="$1"
  local expected_version="$2"
  local python_bin="$3"

  if [[ -x "$venv_dir/bin/python" ]]; then
    local current_version
    current_version="$("$venv_dir/bin/python" -c 'import platform; print(platform.python_version())')"
    if [[ "$current_version" != "$expected_version" ]]; then
      echo "Recreating $venv_dir (expected Python $expected_version, found $current_version)"
      rm -rf "$venv_dir"
    fi
  fi

  if [[ ! -d "$venv_dir" ]]; then
    echo "Creating $venv_dir with Python $expected_version"
    "$python_bin" -m venv "$venv_dir"
  fi
}

install_from_freeze() {
  local venv_dir="$1"
  local freeze_file="$2"
  "$venv_dir/bin/python" -m pip install --upgrade pip
  "$venv_dir/bin/pip" install -r "$freeze_file"
}

resolve_hf_token() {
  if [[ -n "$NOVELQA_HF_TOKEN" ]]; then
    printf '%s' "$NOVELQA_HF_TOKEN"
    return
  fi
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    printf '%s' "$HUGGINGFACE_HUB_TOKEN"
    return
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return
  fi
  printf ''
}

download_novelqa() {
  local token
  token="$(resolve_hf_token)"
  if [[ -z "$token" ]]; then
    echo "NovelQA download requested but no HF token provided." >&2
    echo "Use --hf-token or set HUGGINGFACE_HUB_TOKEN / HF_TOKEN." >&2
    exit 1
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "git is required for NovelQA clone." >&2
    exit 1
  fi

  if [[ -e "$NOVELQA_TARGET" ]]; then
    if [[ "$NOVELQA_FORCE" == "1" ]]; then
      echo "Removing existing NovelQA target: $NOVELQA_TARGET"
      rm -rf "$NOVELQA_TARGET"
    else
      echo "NovelQA target already exists, skipping: $NOVELQA_TARGET"
      return
    fi
  fi

  mkdir -p "$(dirname "$NOVELQA_TARGET")"
  echo "== NovelQA download =="
  echo "Cloning https://huggingface.co/datasets/$NOVELQA_REPO -> $NOVELQA_TARGET"
  git -c "http.extraHeader=Authorization: Bearer ${token}" \
    clone --progress "https://huggingface.co/datasets/$NOVELQA_REPO" "$NOVELQA_TARGET"
}

parse_args "$@"

echo "== Submodules =="
git submodule update --init --recursive

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv is required but not found in PATH." >&2
  echo "Install pyenv and retry." >&2
  exit 1
fi

require_file "$MAIN_FREEZE"
require_file "$LITERARY_FREEZE"

PYENV_ROOT="${PYENV_ROOT:-$(pyenv root)}"

echo "== pyenv Python versions =="
MAIN_PY_BIN="$(ensure_pyenv_python "$MAIN_PY_VERSION" "$PYENV_ROOT")"
LITERARY_PY_BIN="$(ensure_pyenv_python "$LITERARY_PY_VERSION" "$PYENV_ROOT")"

echo "== Virtualenvs =="
ensure_venv "$MAIN_VENV" "$MAIN_PY_VERSION" "$MAIN_PY_BIN"
ensure_venv "$LITERARY_VENV" "$LITERARY_PY_VERSION" "$LITERARY_PY_BIN"

echo "== Installing frozen dependencies =="
install_from_freeze "$MAIN_VENV" "$MAIN_FREEZE"
install_from_freeze "$LITERARY_VENV" "$LITERARY_FREEZE"

echo "== NLTK punkt =="
"$MAIN_VENV/bin/python" - <<'PY'
import nltk
try:
    nltk.data.find("tokenizers/punkt")
    print("punkt already installed")
except LookupError:
    nltk.download("punkt")
PY

echo "== spaCy model =="
"$MAIN_VENV/bin/python" - <<'PY'
import importlib.util, subprocess, sys
model = "en_core_web_sm"
if importlib.util.find_spec(model) is None:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
else:
    print(f"{model} already installed")
PY

if [[ "$DOWNLOAD_NOVELQA" == "1" ]]; then
  download_novelqa
fi

echo ""
echo "✅ Setup complete."
echo "Main env: source .venv/bin/activate"
echo "LiteraryQA env: source .venv-literaryqa/bin/activate"
