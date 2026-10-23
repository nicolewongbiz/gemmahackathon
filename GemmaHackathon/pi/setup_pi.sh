#!/usr/bin/env bash
# Raspberry Pi 5, Raspberry Pi OS 64-bit (Bookworm). Run once, with internet, the night before.
set -e
sudo apt update && sudo apt install -y portaudio19-dev libsndfile1 espeak-ng python3-gpiozero
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv tool install litert-lm || true
mkdir -p models
# Model download (set HF_TOKEN for faster, unthrottled downloads)
uv pip install "huggingface_hub[cli]"
huggingface-cli download litert-community/gemma-4-E2B-it-litert-lm --local-dir models/e2b
# Rename whatever .litertlm file came down to the name voice_agent.py expects:
f=$(ls models/e2b/*.litertlm | head -1); ln -sf "$PWD/$f" models/gemma-4-E2B-it.litertlm
echo "Smoke test:"
litert-lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm --prompt="Say hi in five words" || \
  echo "CLI flags may differ: run 'litert-lm --help'"
echo "Done. Next: python voice_agent.py --text --laptop <laptop-ip>"
