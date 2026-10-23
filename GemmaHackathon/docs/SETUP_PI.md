# Raspberry Pi 5 setup (LiteRT-LM + Gemma 4 E2B)

## Requirements

- Raspberry Pi 5, 8 GB recommended (4 GB reported to work)
- **Raspberry Pi OS 64-bit (Bookworm)** — LiteRT is aarch64-only
- ≥ 10 GB free on the SD card / SSD (E2B bundle ≈ 2.6 GB)
- USB microphone, speaker (USB or 3.5 mm), optional button on GPIO17 → GND
- Active cooler recommended (sustained CPU inference)

## 1. One-shot install (with internet)

```bash
git clone <repo> ~/GemmaHackathon   # or copy the pi/ folder over
cd ~/GemmaHackathon/pi
bash setup_pi.sh
```
The script:
1. `apt install portaudio19-dev libsndfile1 espeak-ng python3-gpiozero`
2. installs `uv`, creates `.venv`, installs `requirements.txt` (`litert-lm-api`, `sounddevice`, `soundfile`, `numpy`, `requests`)
3. `uv tool install litert-lm` (CLI)
4. downloads `litert-community/gemma-4-E2B-it-litert-lm` to `models/e2b/` and links `models/gemma-4-E2B-it.litertlm`
5. runs a CLI smoke test

Manual equivalent:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv && source .venv/bin/activate
uv pip install litert-lm-api sounddevice soundfile numpy requests
uv tool install litert-lm
litert-lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm --prompt="Say hi in five words"
```
If a flag is rejected, run `litert-lm --help` — the tool is in early preview and flags may change.

## 2. Probe the Python API

```bash
python ../scripts/probe_litert.py --model models/gemma-4-E2B-it.litertlm --audio
```
Confirms how audio content is passed. Fix `pi/litert_adapter.py` if needed (`# VERIFY ON SITE:`).

## 3. Audio check

```bash
arecord -l                      # list mics
aplay -l                        # list speakers
espeak-ng "hello from the pi"
python -c "import sounddevice as sd; print(sd.query_devices())"
```
Set the default devices in `~/.asoundrc` or pass `--mic <index>` to the voice agent.

## 4. Run the voice agent

```bash
source .venv/bin/activate
python voice_agent.py --text --laptop 192.168.50.1      # typed commands, debug first
python voice_agent.py --laptop 192.168.50.1             # push-to-talk (ENTER)
python voice_agent.py --gpio --laptop 192.168.50.1      # hardware button on GPIO17
python voice_agent.py --stt vosk --laptop 192.168.50.1  # fallback if E2B audio is slow
```
Each command prints Gemma latency. Target < 4 s from button release to intent.

## 5. Network

Static IP on `eth0` (NetworkManager on Bookworm):
```bash
sudo nmcli con add type ethernet ifname eth0 con-name hack ipv4.method manual ipv4.addresses 192.168.50.2/24
sudo nmcli con up hack
ping 192.168.50.1
curl http://192.168.50.1:8000/status
```

## 6. Autostart on boot (optional)

```bash
sudo cp voice_agent.service /etc/systemd/system/
sudo systemctl enable --now voice_agent
journalctl -u voice_agent -f
```

## Performance notes

- Google reports Gemma 4 E2B on Pi 5 via LiteRT-LM: ~99 tok/s prefill, ~9 tok/s decode, ~1.4 GB peak RAM (4 threads, CPU).
- Keep the system prompt short and the reply JSON short — decode speed dominates latency.
- GPU on the Pi is experimental (Vulkan/V3DV); use CPU.
- Fallback for intents: Gemma 3 270M (text) + Vosk STT.
