# CLAUDE.md — Show Me Once

Hackathon project (Google Gemma Hardware Hackathon, Oct 24 2026). A Kinova arm learns a sorting rule from one human demo and repeats it. Everything on-device; no cloud calls ever.

## Devices
- Laptop (RTX 4070, 8 GB): `main.py` FastAPI, perception, Gemma 4 E4B planner, Kortex arm control.
- Raspberry Pi 5: `pi/voice_agent.py`, Gemma 4 E2B via LiteRT-LM, audio in → intent JSON.

## Run in mock mode (no hardware)
```
SMO_USE_MOCK_ARM=1 SMO_GEMMA_BACKEND=fake uvicorn main:app --port 8000
python -m scripts.demo_cli
pytest -q
```

## Hard rules
1. Gemma never outputs coordinates — it picks numbered marks and bin A/B. Geometry is deterministic.
2. Every Gemma call returns one JSON object; extract, validate, whitelist ids, retry ×2.
3. All `litert_lm` usage lives in `planner/litert_adapter.py` and `pi/litert_adapter.py` only.
4. Every hardware dependency has a mock (MockArm, FakeCamera, FakeBackend). Tests must pass without hardware.
5. Safety: bounds-check every arm target, `stop` interrupts immediately, speed limited.
6. All tunables in `config.py`, overridable via `SMO_<NAME>` env vars.

## Known unknowns (`# VERIFY ON SITE:`)
- LiteRT-LM `Engine` kwargs for GPU / vision / audio and multimodal content format → `scripts/probe_litert.py`.
- Kortex wheel version, Gen3 vs Gen3 lite gripper values.
- `TABLE_TAGS`, `Z_GRASP`, `TOOL_ORIENT` — measured on site.

## Docs
`docs/ARCHITECTURE.md`, `docs/API.md`, `docs/PROMPTS.md`, setup guides, troubleshooting. Keep them in sync with code changes.
