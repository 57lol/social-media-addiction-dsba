#!/bin/bash
# Double-click to launch the REST API (FastAPI). Creates the environment on first run.
cd "$(dirname "$0")" || exit 1
if [ ! -x ".venv/bin/python" ]; then
  echo "🌸 First run: setting up the environment (a few minutes)…"
  python3 -m venv .venv \
    && ./.venv/bin/python -m pip install -q --upgrade pip \
    && ./.venv/bin/python -m pip install -q -r requirements.txt
fi
echo "🔌 Starting REST API → docs at http://localhost:8000/docs  (Ctrl+C to stop)"
exec ./.venv/bin/python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
