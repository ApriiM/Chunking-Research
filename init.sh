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
    echo "Installing Python $version with pyenv"
    pyenv install -s "$version"
  fi

  local py_bin="$pyenv_root/versions/$version/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "pyenv python not found for $version: $py_bin" >&2
    exit 1
  fi
  echo "$py_bin"
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

echo ""
echo "✅ Setup complete."
echo "Main env: source .venv/bin/activate"
echo "LiteraryQA env: source .venv-literaryqa/bin/activate"
