# Show Me Once

**A robot arm that learns a chore from one demonstration, explains what it learned, and does it — 100% on-device with Google Gemma.**

Built for the Google Gemma Hardware Hackathon · AGI House · Oct 24, 2026
Track: Physical AI, IoT & Robotics · Also targeting: Best Use of LiteRT, Best Use of Raspberry Pi, Qualcomm hardware awards

Proposal: https://claude.ai/artifact/76eB16jTgB2qvBWz5kRDUt

---

## What it does

1. **Teach** — say "watch me" and sort a few objects into bin A and bin B.
2. **Explain** — Gemma 4 (vision) infers the rule and says it out loud: *"Food goes in A, tools go in B."*
3. **Do** — dump new objects on the table, say "go". The Kinova arm sorts them, narrating each choice.
4. **Correct** — "actually, red things go in B." Gemma rewrites the rule; the arm re-sorts.
5. **Offline** — pull the network. Nothing changes. No cloud anywhere in the loop.

## How it works (one paragraph)

A **Raspberry Pi 5** runs **Gemma 4 E2B on LiteRT-LM** as the voice interface: microphone audio goes straight into the model (no separate speech-to-text), which returns a JSON intent sent to the laptop. The **laptop (RTX 4070)** runs **Gemma 4 E4B on LiteRT-LM** (Ollama fallback) as the planner. A webcam sees the table; **AprilTags** give a pixel→metre homography; objects are found by diffing against an empty-table snapshot and drawn as **numbered marks**. Gemma only picks *which number goes in which bin* — deterministic code turns that into a **Kinova Kortex** pick-and-place. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Hardware

| Item | Role |
|---|---|
| Laptop, NVIDIA RTX 4070 (8 GB) | Vision planner (Gemma 4 E4B), perception, arm control, HTTP server |
| Raspberry Pi 5 (8 GB), USB mic, speaker | Voice agent (Gemma 4 E2B via LiteRT-LM), espeak-ng |
| Kinova Gen3 / Gen3 lite | Manipulator (Kortex API) |
| USB webcam, top-down | Scene camera |
| AprilTags tag36h11, 5 cm: IDs 0–3, 10, 11 | Table corners + bin markers |
| Ethernet cable | Offline Pi ↔ laptop link |

## Quickstart

**Mock mode (no robot, no Pi, no model) — try this first:**
```bash
pip install -r requirements-laptop.txt
SMO_USE_MOCK_ARM=1 SMO_GEMMA_BACKEND=fake uvicorn main:app --port 8000
# open http://localhost:8000  → use the buttons, or:
python -m scripts.demo_cli
```

**Real hardware:**
1. Laptop setup → [docs/SETUP_LAPTOP.md](docs/SETUP_LAPTOP.md)
2. Pi setup → [docs/SETUP_PI.md](docs/SETUP_PI.md)
3. Kinova + calibration → [docs/KINOVA_AND_CALIBRATION.md](docs/KINOVA_AND_CALIBRATION.md)
4. Run:
   ```bash
   # laptop
   uvicorn main:app --host 0.0.0.0 --port 8000
   # pi
   python pi/voice_agent.py --laptop 192.168.50.1
   ```
5. Say: **"table is empty"** → **"watch me"** → *(demo)* → **"done"** → **"go"**

## Repository layout

```
config.py            every tunable (env override: SMO_<NAME>)
main.py              laptop FastAPI app + state machine + sort loop
common/              JSON extraction, schemas
planner/             LiteRT adapter, backends (litert/ollama/fake), prompts, Brain
perception/          camera, AprilTag calibration, object finder + marks, keyframes
robot/               Arm protocol, MockArm, KinovaArm, pick_and_place
pi/                  voice agent, LiteRT adapter, setup script, systemd unit
scripts/             model download, probes, test_gemma, test_click_pick, test_kinova, print_tags, demo_cli
tests/               pytest suite — runs with no hardware
docs/                all documentation
CLAUDE_CODE_PROMPT.md  one-shot prompt to (re)build the full codebase with Claude Code
```

> The starter code in this folder is a skeleton. Run `CLAUDE_CODE_PROMPT.md` through Claude Code in this directory to build out the full tree above.

## Documentation

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data flow, state machine, design rules |
| [SETUP_LAPTOP.md](docs/SETUP_LAPTOP.md) | Python env, LiteRT-LM GPU, Ollama fallback, models |
| [SETUP_PI.md](docs/SETUP_PI.md) | Pi OS, LiteRT-LM, Gemma E2B, audio, networking, autostart |
| [KINOVA_AND_CALIBRATION.md](docs/KINOVA_AND_CALIBRATION.md) | Kortex install, network, tags, homography, grasp tuning |
| [API.md](docs/API.md) | HTTP endpoints, intent schema, Gemma JSON contracts |
| [PROMPTS.md](docs/PROMPTS.md) | Prompt design, set-of-marks, JSON contracts |
| [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | Day-of timeline, demo script, pitch, judging Q&A |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Known failure modes and fixes |
| [CHECKLIST.md](docs/CHECKLIST.md) | Night-before and day-of checklists |

## Team

| Role | Owns |
|---|---|
| Arm + calibration | `robot/`, `perception/calibrate.py` |
| Gemma planner | `planner/`, `main.py` |
| Pi + LiteRT | `pi/` |
| Demo + pitch | `docs/DEMO_SCRIPT.md`, table setup, backup video |

## License

MIT for code written at the event. Gemma models are subject to the Gemma terms of use.
