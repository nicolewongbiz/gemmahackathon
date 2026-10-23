# Laptop setup (RTX 4070)

Do this the night before, with internet. Everything after is offline.

## 1. Python environment

```bash
cd GemmaHackathon
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install -r requirements-laptop.txt
```

Requires Python 3.10+ and a recent NVIDIA driver.

## 2. Gemma 4 E4B for LiteRT-LM

```bash
bash scripts/download_models.sh        # Windows: powershell scripts/download_models.ps1
```
This downloads `litert-community/gemma-4-E4B-it-litert-lm` from Hugging Face into `data/` and copies the `.litertlm` to `data/gemma-4-E4B-it.litertlm`. Setting `HF_TOKEN` gives faster, unthrottled downloads.

CLI check (optional):
```bash
uv tool install litert-lm
litert-lm run --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm --backend=gpu --prompt="Say hi"
```

## 3. Probe the Python API (important)

The exact `litert_lm.Engine` arguments for GPU / vision were not confirmed before the event.
```bash
python -m scripts.probe_litert
```
It prints the version, `Engine` signature, `Backend` members, content helpers, and runs text + image tests. If the image test fails, fix the construction order in `planner/litert_adapter.py` (marked `# VERIFY ON SITE:`).

## 4. Ollama fallback

Install Ollama, then:
```bash
ollama pull gemma4:e4b        # verify the exact tag with `ollama list` / ollama.com
SMO_GEMMA_BACKEND=ollama python -m scripts.test_gemma
```
Keep LiteRT-LM on the Pi even if the laptop falls back — the project still qualifies for the LiteRT prize.

## 5. Webcam check

```bash
python -m scripts.test_gemma --backend litert
```
Prints the objects Gemma sees and latency. Target: < 3 s per call. If slower, lower `PROMPT_IMAGE_MAX` to 512.

## 6. Network for the Pi

Plug the Ethernet cable between laptop and Pi. Give the laptop a static IP on that interface, e.g. `192.168.50.1/24`, and the Pi `192.168.50.2/24`. (The Kinova is on a separate interface/USB-Ethernet at `192.168.1.11/24`.)

Allow inbound TCP 8000 in the laptop firewall (Windows: Defender Firewall → inbound rule).

## 7. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 for the live view and backup buttons.

## Environment overrides

Any `config.py` value can be overridden with `SMO_<NAME>`, e.g.:
```bash
SMO_USE_MOCK_ARM=0 SMO_GEMMA_BACKEND=ollama SMO_CAMERA_INDEX=1 uvicorn main:app --host 0.0.0.0
```
