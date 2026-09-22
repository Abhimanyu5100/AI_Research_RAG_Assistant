#!/usr/bin/env bash
# Run the FastAPI backend and the Next.js frontend together.
#
#   ./scripts/dev.sh
#
# Backend  -> http://localhost:8000
# Frontend -> http://localhost:3000  (proxies /api/* to the backend)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "rag/bin/python" ]]; then PY="rag/bin/python"; else PY="python3"; fi
fi

if [[ ! -d "web/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd web && npm install)
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Starting backend on :8000 ..."
"$PY" -m uvicorn src.api.main:app --port 8000 --reload &

echo "Starting frontend on :3000 ..."
(cd web && npm run dev) &

wait
