#!/usr/bin/env bash
# One-shot setup for Linux/macOS:  bash setup.sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
if ls kortex_api*.whl >/dev/null 2>&1; then
  .venv/bin/pip install kortex_api*.whl && .venv/bin/pip install "protobuf==3.20.3"
else
  echo "No kortex_api*.whl here: download from Kinova Artifactory, drop it in this folder, re-run."
fi
[ -f models/gemma.litertlm ] || .venv/bin/python scripts/download_model.py
.venv/bin/python scripts/selftest.py
echo "Done. Next: ./run.sh --mock   or   ./run.sh --real"
