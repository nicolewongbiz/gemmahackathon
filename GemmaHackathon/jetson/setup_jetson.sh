#!/usr/bin/env bash
# Jetson Orin Nano Super, JetPack 6.x. Run once, with internet, the night before:
#   cd ~/GemmaHackathon && bash jetson/setup_jetson.sh
set -e
cd "$(dirname "$0")/.."
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.11 .venv
uv pip install -p .venv -r jetson/requirements.txt
# Gemma on the Jetson GPU. The Ollama installer detects JetPack and sets up CUDA + a systemd service.
command -v ollama || curl -fsSL https://ollama.com/install.sh | sh
MODEL=$(.venv/bin/python -c "import config; print(config.EYES_GEMMA_MODEL)")
ollama pull "$MODEL"
.venv/bin/python -c "import cv2, pupil_apriltags, fastapi; print('python deps ok')"
echo "Smoke test:"
ollama run "$MODEL" "Say hi in five words" || true
echo "Power mode (switch to MAXN SUPER for the demo with: sudo nvpmodel -m <id>):"
sudo nvpmodel -q || true
echo "Done. Next: .venv/bin/uvicorn jetson.eyes_server:app --host 0.0.0.0 --port 8001"
