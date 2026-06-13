#!/bin/bash
# Double-click to launch the web report (Streamlit) in your browser.
# On first run it creates a local environment automatically (takes a few minutes).
cd "$(dirname "$0")" || exit 1
if [ ! -x ".venv/bin/python" ]; then
  echo "🌸 First run: setting up the environment (a few minutes)…"
  python3 -m venv .venv \
    && ./.venv/bin/python -m pip install -q --upgrade pip \
    && ./.venv/bin/python -m pip install -q -r requirements.txt
fi
echo "🌸 Starting the web report → http://localhost:8501  (Ctrl+C to stop)"
exec ./.venv/bin/python -m streamlit run app/streamlit_app.py
