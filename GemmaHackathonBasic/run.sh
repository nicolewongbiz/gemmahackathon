#!/usr/bin/env bash
# ./run.sh --mock | --real  [--backend litert|ollama|rules]
cd "$(dirname "$0")"
.venv/bin/python app.py "$@"
