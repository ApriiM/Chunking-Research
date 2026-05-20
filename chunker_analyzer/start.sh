#!/usr/bin/env bash
PYTHON=python3.12
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BACKEND_PORT="${BACKEND_PORT:-8000}"
SIM_PORT="${SIM_PORT:-5051}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

echo "============================================================"
echo " Project startup"
echo " Backend   : $BACKEND_PORT"
echo " Similarity: $SIM_PORT"
echo " Frontend  : $FRONTEND_PORT"
echo "============================================================"

# check uv
if ! $PYTHON -m uv --version &>/dev/null; then
  echo "ERROR: uv not installed -> pip install uv"
  exit 1
fi

# =========================
# BACKEND
# =========================
echo "[1/3] Backend..."

cd "$SCRIPT_DIR/backend"
$PYTHON -m uv sync

DATA_ROOT="${DATA_ROOT:-data}" PORT="$BACKEND_PORT" \
$PYTHON -m uv run python app.py &

BACKEND_PID=$!

# =========================
# SIMILARITY BACKEND
# =========================
echo "[2/3] Similarity backend..."

cd "$SCRIPT_DIR/similarity_backend"
$PYTHON -m uv sync --extra need_torch

SIM_PORT="$SIM_PORT" \
$PYTHON -m uv run python app.py &

SIM_PID=$!

# =========================
# FRONTEND
# =========================
echo "[3/3] Frontend..."

cd "$SCRIPT_DIR/frontend"

npm install --silent
npx vite --port "$FRONTEND_PORT" &

FRONTEND_PID=$!

# =========================
# CLEANUP
# =========================
cleanup() {
  echo ""
  echo "Stopping..."
  kill $BACKEND_PID $SIM_PID $FRONTEND_PID 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM

echo ""
echo "============================================================"
echo " Backend   : http://localhost:$BACKEND_PORT"
echo " Similarity: http://localhost:$SIM_PORT"
echo " Frontend  : http://localhost:$FRONTEND_PORT"
echo "============================================================"

wait