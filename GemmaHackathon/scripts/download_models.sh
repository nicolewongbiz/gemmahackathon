#!/usr/bin/env bash
# Laptop: fetch Gemma 4 E4B LiteRT bundle into data/. Run from repo root.
set -e
pip install "huggingface_hub[cli]"
huggingface-cli download litert-community/gemma-4-E4B-it-litert-lm --local-dir data/e4b
f=$(ls data/e4b/*.litertlm | head -1); cp "$f" data/gemma-4-E4B-it.litertlm
echo "Fallback: ollama pull gemma4:e4b   (verify tag on ollama.com)"
