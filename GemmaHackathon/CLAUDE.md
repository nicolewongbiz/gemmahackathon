# CLAUDE.md — Show Me Once

Hackathon project (Gemma Hardware Hackathon, Oct 24 2026). A Kinova Gen3 learns a sorting rule from one human demo and repeats it. Everything runs on-device; no cloud calls.

`README.md` is the single source of documentation: setup, calibration, run, demo, troubleshooting. Keep it in sync with code changes, and don't add separate doc files.

## Devices
- Laptop: `main.py` FastAPI, perception, Gemma 4 E4B planner, Kortex arm control.
- Camera: USB webcam on a stand above the table (our arm has no Kinova Vision module). `config.CAMERA` picks webcam / kinova (RTSP) / mock. Always call `arm.look()` before a perception frame: it parks the arm out of view (or at the viewing pose if `CAMERA_ON_WRIST`).
- Raspberry Pi 5: `pi/voice_agent.py`, Gemma 4 E2B via LiteRT-LM, audio in → intent JSON.

## Without hardware
Set `USE_MOCK_ARM = True` and `CAMERA = "mock"` in `config.py`. Gemma still needs a backend (LiteRT model or Ollama).

## Hard rules
1. Gemma never outputs coordinates. It picks numbered marks and bin A/B, and geometry is deterministic.
2. Every Gemma call returns one JSON object: extract it, whitelist ids, retry ×2.
3. `litert_lm` usage lives only in `planner/gemma_vision.py` and `pi/voice_agent.py`.
4. All tunables live in `config.py`.
5. Safety: slow moves, `stop` interrupts immediately.

## Measured on site
`TABLE_TAGS`, `LOOK_POSE`, `Z_GRASP`, `TOOL_ORIENT`, `GRIPPER_CLOSE_SPEED`. The laptop LiteRT config (GPU text + CPU vision, `gemma-4-E4B-it.litertlm`) is verified on Windows; the Pi's audio path is not.
