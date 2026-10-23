# One-shot build prompt for Claude Code

Copy everything below the line into Claude Code, run from inside the `GemmaHackathon/` project directory.

---

You are building **Show Me Once**, a hackathon project for the Google Gemma Hardware Hackathon at AGI House (Oct 24, 2026, Physical AI / Robotics track, also targeting Best Use of LiteRT and Best Use of Raspberry Pi). Build out every file needed for a working, demo-ready system in this directory. A starter skeleton may already exist here — read every existing file first, keep what is correct, fix what is wrong, and fill in everything missing. Do not delete `docs/` content or `README.md`; update them if your changes make them stale.

## 1. What the system does

A Kinova robot arm learns a tabletop sorting chore from ONE human demonstration, explains the rule it inferred out loud, then sorts new objects by that rule and accepts spoken corrections. Everything runs on-device with zero cloud calls — this is a hard rule of the hackathon.

Demo flow the code must support end to end:
1. "Table is empty" → laptop snapshots the empty table (background for object detection).
2. "Watch me" → laptop records keyframes while a human sorts 4–6 objects into bin A and bin B.
3. "Done" → Gemma (vision) looks at the keyframes and returns the rule as JSON; the Pi speaks it ("Food goes in A, tools go in B").
4. "Go" → loop: detect objects, draw numbered marks, Gemma picks one object number + bin, geometry converts pixel → robot metres, arm picks and places, speaks "banana, bin A". Repeat until empty or "stop".
5. "Actually, red things go in B" → Gemma rewrites the rule; next "go" uses it.
6. "Stop" at any time halts the arm immediately.

## 2. Hardware

- **Laptop:** Windows or Linux, NVIDIA RTX 4070 laptop GPU (8 GB VRAM). Runs perception, Gemma 4 E4B vision planner, Kinova control, HTTP server.
- **Raspberry Pi 5** (assume 8 GB, Raspberry Pi OS 64-bit Bookworm): runs Gemma 4 E2B via LiteRT-LM as the voice front end. USB mic + USB/3.5mm speaker. Optional push button on GPIO17.
- **Kinova arm:** Gen3 or Gen3 lite (unconfirmed — support both; they share the Kortex API). Default IP 192.168.1.10, admin/admin, laptop on static 192.168.1.11/24.
- **Webcam:** USB, mounted looking down at the table, 1280×720.
- **AprilTags:** family tag36h11, 5 cm. IDs 0–3 at table corners (positions measured in robot base frame), IDs 10 and 11 on bin A and bin B.
- Pi ↔ laptop over direct Ethernet or offline hotspot. No internet at demo time.

## 3. Hard design rules

1. **Gemma never outputs coordinates.** It only chooses among numbered marks drawn on the image ("set-of-marks" prompting) and chooses bin "A" or "B". All geometry (homography, pick heights, bin positions) is deterministic code.
2. **Every Gemma call returns a single JSON object.** Always parse robustly (extract first `{...}`, retry up to 2× with "Reply with ONLY valid JSON"), validate against the expected schema, and reject object ids not in the current detected set.
3. **LiteRT-LM is the primary runtime on both devices** (`pip install litert-lm-api`, `import litert_lm`). The laptop must also support an Ollama fallback (`/api/chat`, `format: "json"`, base64 images) selected by one config flag, because LiteRT-LM vision on the laptop GPU may not work on the day.
4. **Mock everything that is hardware.** `USE_MOCK_ARM`, and a `--fake-camera <folder of jpgs>` option, so the whole pipeline runs on a laptop with no robot and no webcam. A `FakeGemma` backend that returns canned JSON so tests run with no model at all.
5. **Safety:** arm speed limited, workspace bounds checked before every move (reject any target outside the table rectangle defined by tags 0–3 plus a margin, or z below `Z_GRASP`), `stop` intent calls `base.Stop()` immediately from any thread, and the sort loop checks a stop flag between every motion.
6. **One config file** (`config.py`) for every tunable. No magic numbers elsewhere.
7. Python 3.10+, type hints, docstrings, `logging` (not print) with timestamps. Keep dependencies minimal.

## 4. LiteRT-LM API — what is known vs. unknown

Known (from Google's LiteRT-LM Python docs):
- `litert_lm.Engine(<path to .litertlm>)` loads a model; use as a context manager.
- `engine.create_conversation(messages=[{"role": "system", "content": "..."}], tools=[...])` → conversation (context manager).
- `conv.send_message(str | message)` → response dict; text at `response["content"][0]["text"]`.
- `conv.send_message_async(...)` → iterator of chunk dicts (streaming).
- Tools = plain Python functions with type hints + docstrings, Gemma 4 supports tool calling.
- Multimodal inputs exist (vision + audio need the engine configured with a vision / audio backend); the docs mention `litert_lm.Contents.of(...)` for mixed content.
- CLI: `uv tool install litert-lm`, then `litert-lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm --prompt="..."` (E4B: `litert-community/gemma-4-E4B-it-litert-lm`, supports `--backend=gpu`).

Unknown — you must NOT hard-code a guess in one place and hope:
- Exact `Engine(...)` kwarg names for GPU backend, vision backend, audio backend.
- Exact content format for images and audio inside a message.

So: put ALL LiteRT-LM calls behind one adapter module per device (`planner/litert_adapter.py`, `pi/litert_adapter.py`, or one shared `common/litert_adapter.py` copied to the Pi). The adapter must:
- At init, introspect `litert_lm` (`inspect.signature(litert_lm.Engine)`, `hasattr(litert_lm, "Backend")`, `hasattr(litert_lm, "Contents")`) and try a short ordered list of known-plausible constructions, logging which one succeeded.
- Expose exactly: `chat(system: str, text: str, image_paths: list[str] = [], audio_path: str | None = None) -> str`.
- Provide `scripts/probe_litert.py` that prints the installed version, `Engine` signature, available `Backend` members, `Contents` helpers, and runs a text-only hello + one image + one audio test, so the team can fix the adapter in minutes on site.

## 5. File tree to produce

```
GemmaHackathon/
├── CLAUDE.md                     # short project context for future Claude Code sessions
├── README.md                     # keep/update
├── config.py                     # all tunables (see §6)
├── main.py                       # laptop FastAPI app + orchestration
├── requirements-laptop.txt
├── pyproject.toml                # optional; ruff + pytest config
├── .gitignore
├── common/
│   ├── __init__.py
│   ├── jsonutil.py               # extract_json(), validate(schema)
│   └── schemas.py                # dataclasses / TypedDicts for Intent, TeachResult, PlanStep, CorrectResult
├── planner/
│   ├── __init__.py
│   ├── litert_adapter.py         # LiteRT-LM wrapper (see §4)
│   ├── backends.py               # LiteRTBackend, OllamaBackend, FakeBackend  -> .ask(system, text, images)
│   ├── prompts.py                # SYSTEM, TEACH, PLAN, CORRECT templates
│   └── brain.py                  # Brain: teach(), plan(), correct(); persists rule to data/rule.json
├── perception/
│   ├── __init__.py
│   ├── camera.py                 # threaded webcam + FakeCamera(folder)
│   ├── calibrate.py              # TableMap: AprilTag homography, bins, tag mask, bounds check; __main__ = live calib viewer
│   ├── objects.py                # ObjectFinder (empty-table diff), draw_marks(), KeyframeRecorder
│   └── debug_view.py             # optional OpenCV window showing marks, chosen object, bins, state
├── robot/
│   ├── __init__.py
│   ├── arm.py                    # Arm protocol, MockArm, KinovaArm (Kortex), make_arm()
│   └── motions.py                # pick_and_place() with bounds checks + stop checks
├── pi/
│   ├── README.md                 # Pi-only quickstart
│   ├── requirements.txt
│   ├── setup_pi.sh               # one-shot install + model download + smoke test
│   ├── litert_adapter.py         # same contract as planner/litert_adapter.py (copy, Pi-tuned: CPU)
│   ├── voice_agent.py            # push-to-talk → Gemma E2B → intent JSON → POST laptop; poll /say → espeak-ng
│   └── voice_agent.service       # systemd unit so it starts on boot
├── scripts/
│   ├── __init__.py
│   ├── download_models.sh        # laptop: E4B .litertlm into data/
│   ├── download_models.ps1       # same for Windows PowerShell
│   ├── probe_litert.py           # see §4
│   ├── test_gemma.py             # webcam frame → Gemma → JSON, prints latency, --backend flag
│   ├── test_click_pick.py        # click pixel → arm picks → bin A (milestone 1)
│   ├── test_kinova.py            # connect, print pose, open/close gripper, move 5 cm up and back
│   ├── print_tags.py             # writes a printable PDF/PNG sheet of tag36h11 IDs 0-3,10,11 at 5 cm (generate tags with OpenCV aruco DICT_APRILTAG_36h11)
│   └── demo_cli.py               # keyboard driver: e/t/d/g/s/c keys send intents to the laptop (no Pi needed)
├── tests/
│   ├── test_jsonutil.py
│   ├── test_brain_fake.py        # Brain with FakeBackend: teach → plan → correct
│   ├── test_homography.py        # synthetic tag positions → known px→m mapping within 1 mm
│   ├── test_objects.py           # synthetic images with drawn rectangles → correct count + centroids
│   └── test_api.py               # FastAPI TestClient with mock arm + fake camera + fake gemma: full demo flow
├── data/.gitkeep
└── docs/                         # keep existing docs, update if needed
```

## 6. `config.py` contents (at minimum)

Network (`LAPTOP_HOST`, `LAPTOP_PORT=8000`), camera (`CAMERA_INDEX`, `FRAME_W/H`, `PROMPT_IMAGE_MAX=768`), Gemma (`GEMMA_BACKEND in {"litert","ollama","fake"}`, `LITERT_MODEL` path, `LITERT_HF_REPO`, `OLLAMA_URL`, `OLLAMA_MODEL="gemma4:e4b"`, `GEMMA_TEMPERATURE=0.2`, `GEMMA_TIMEOUT_S`), tags (`TABLE_TAGS: dict[int, tuple[float,float]]` in metres, `BIN_TAGS={10:"A",11:"B"}`, `TAG_SIZE_M=0.05`, `WORKSPACE_MARGIN_M`), robot (`ROBOT_IP/USER/PASS`, `USE_MOCK_ARM`, `ARM_MODEL in {"gen3","gen3_lite"}`, `Z_HOVER`, `Z_GRASP`, `Z_TABLE_OFFSET`, `TOOL_ORIENT` (deg), `GRIPPER_OPEN/CLOSED`, `BIN_DROP` fallback xyz, `MAX_LINEAR_SPEED`), object detection (`DIFF_THRESHOLD`, `MIN_AREA_PX`), teaching (`KEYFRAME_MAX=6`, `STILL_FRAMES`, `CHANGE_THRESH`), loop (`MAX_PICKS=12`). Allow overrides via environment variables (`SMO_<NAME>`) so nobody edits code during the demo.

## 7. Module specs

**`perception/camera.py`** — `Camera(index)` reads in a daemon thread, `.read()` returns a copy of the newest frame, `.close()`. `FakeCamera(folder)` cycles through JPGs. Factory `make_camera()` chooses from config/CLI.

**`perception/calibrate.py`** — `TableMap.update(frame) -> bool` detects tags with `pupil_apriltags.Detector(families="tag36h11")`, computes `cv2.findHomography` from tag centres (px) to `TABLE_TAGS` (m) when all four corner tags are visible, caches the last good H (keep using it if tags are temporarily occluded by the arm), records bin pixel centres. `px_to_table(u,v)`, `bin_xy(name)` (tag if seen, else `BIN_DROP`), `tag_mask(frame)` (dilated tag polygons, so tags are never proposed as objects), `in_bounds(x,y)`. Running as `python -m perception.calibrate` opens a live window: tag ids overlaid, CALIBRATED/NOT status, click a pixel → logs table metres.

**`perception/objects.py`** — `ObjectFinder.snapshot_empty(frame)`; `find(frame, ignore_mask) -> dict[int, (u,v)]` via blurred absdiff → threshold → open/close morphology → contours ≥ `MIN_AREA_PX` → centroids, ids 1..N in stable reading order. Also mask out the arm region if the arm is over the table (simple: only detect when the arm is at home). `draw_marks(frame, objs, bins_px)` draws white filled circles with black numbers (large, legible to a VLM) and "BIN A"/"BIN B" labels. `KeyframeRecorder.push(frame)` saves a frame each time the scene changes then settles for `STILL_FRAMES`; `.result(max)` evenly subsamples.

**`planner/prompts.py`** — templates:
- `SYSTEM`: robot planner, top-down images, numbered white circles = objects, bins A/B, answer one JSON object only.
- `TEACH`: N ordered frames of a person sorting → `{"rule": str, "bins": {"A": str, "B": str}, "say": str}`.
- `PLAN`: rule + bin descriptions + list of valid ids → `{"object": int|null, "label": str, "bin": "A"|"B", "reason": str, "say"?: str}`.
- `CORRECT`: current rule + user's words → same shape as TEACH.
Include 1 short few-shot example in each prompt to stabilise the JSON.

**`planner/backends.py`** — common interface `ask(system, text, images: list[np.ndarray]) -> str`. Images are resized so the longest side ≤ `PROMPT_IMAGE_MAX` and written to `data/_prompt_<i>.jpg` (LiteRT) or base64 (Ollama). `FakeBackend` returns deterministic canned JSON, alternating bins, choosing the lowest valid id (parse ids from the prompt).

**`planner/brain.py`** — `Brain(backend)`: `teach(frames) -> say`, `plan(marked_frame, ids) -> PlanStep`, `correct(text) -> say`. Persists rule to `data/rule.json`, loads on start. Logs every prompt/response with latency to `data/gemma_log.jsonl` (great for debugging and for the pitch: "median 1.8 s per decision").

**`robot/arm.py`** — `Arm` Protocol: `move_to(x,y,z)`, `gripper(v)`, `home()`, `stop()`, `get_pose()`, `close()`. `MockArm` logs and sleeps. `KinovaArm` uses Kortex:
```python
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2
```
TCP port 10000, CreateSessionInfo(username, password, session_inactivity_timeout=60000, connection_inactivity_timeout=2000), `SetServoingMode(SINGLE_LEVEL_SERVOING)`, cartesian moves via `Action.reach_pose.target_pose` + `OnNotificationActionTopic` waiting for `ACTION_END`/`ACTION_ABORT` with timeout, gripper via `GripperCommand(mode=GRIPPER_POSITION)` finger value 0..1, home via `ReadAllActions(REACH_JOINT_ANGLES)` named "Home", `stop()` → `base.Stop()`, `get_pose()` via `BaseCyclicClient.RefreshFeedback().base.tool_pose_*`. Apply speed limits via `reach_pose.constraint.speed.translation` if available. Import kortex lazily so the rest runs without the wheel installed.

**`robot/motions.py`** — `pick_and_place(arm, xy, bin_xy, stop_flag)`: open → hover → descend → close → lift → move above bin → open → hover; bounds check every target; check `stop_flag` between steps and abort cleanly.

**`main.py`** (FastAPI, run with `uvicorn main:app --host 0.0.0.0 --port 8000`) — state machine `idle | teaching | thinking | sorting`. Endpoints:
- `POST /intent` body `{"intent": "empty|teach|done|go|stop|correct", "text": ""}` → returns `{"ok", "mode", "rule"}` quickly; long work happens in a background thread (never block the request > 1 s, except `stop` which must act instantly).
- `GET /say` → `{"say": [str,...]}` drains the speech queue (Pi polls every 0.5 s).
- `GET /status` → mode, rule, calibrated, last plan step, last latency, objects visible.
- `GET /frame.jpg` → latest marked debug frame (so a browser on any device can show what Gemma sees — nice for judges).
- `GET /` → tiny HTML page: live `/frame.jpg` refresh, status, and buttons for each intent (backup control surface if the Pi fails).
CLI flags via env/config: `--mock-arm`, `--fake-camera DIR`, `--backend`.

**`pi/voice_agent.py`** — loads Gemma E2B once via `pi/litert_adapter.py` (CPU). Trigger: ENTER key, or GPIO17 button with `gpiozero` if `--gpio`. Records 4 s mono 16 kHz WAV with `sounddevice`. Modes `--stt gemma` (default: audio goes straight into E2B) or `--stt vosk` (offline fallback, model in `pi/models/vosk-model-small-en-us-0.15`) or `--text` (typed). System prompt maps speech → `{"intent": "empty|teach|done|go|stop|correct|chat", "text": "<words>", "reply": "<short spoken reply>"}`. Fast path: if the transcript/keyword clearly says "stop", send stop immediately without waiting for Gemma. Background thread polls `/say` and speaks with `espeak-ng -s 165`. Prints latency per command. Handles laptop unreachable gracefully ("I can't reach the laptop").

**`pi/setup_pi.sh`** — apt (`portaudio19-dev libsndfile1 espeak-ng python3-gpiozero`), uv, venv, requirements, `uv tool install litert-lm`, `huggingface-cli download litert-community/gemma-4-E2B-it-litert-lm --local-dir models/e2b`, symlink the `.litertlm` to `models/gemma-4-E2B-it.litertlm`, CLI smoke test, optional Vosk model download, install the systemd unit.

## 8. Tests and acceptance criteria

Run `pytest -q`; all must pass with no GPU, no robot, no camera, no model (use FakeBackend, FakeCamera, MockArm). Specifically:
1. `extract_json` handles code fences, leading prose, trailing junk, and invalid JSON (raises).
2. Homography test: 4 synthetic tag centres → `px_to_table` error < 1 mm at 10 random points.
3. Object finder on synthetic images finds exactly the drawn objects, ignores masked tag areas.
4. Brain with FakeBackend: teach returns a rule; plan never returns an id outside the given set; correct changes the rule.
5. API flow test: empty → teach → (push fake frames) → done → go → sort loop runs MockArm through ≥ 1 pick → stop halts. `/say` returns the spoken lines in order.
6. Bounds: a target outside the workspace raises and the arm never receives the move.

Also make sure: `python -m perception.calibrate`, `python -m scripts.test_gemma --backend fake`, `python -m scripts.demo_cli`, and `uvicorn main:app` all start without errors in mock mode. Run `ruff` (or at least `python -m py_compile` on everything) and fix all errors.

## 9. Process

1. Read all existing files and `docs/`. Make a short plan.
2. Build `config.py`, `common/`, then perception, planner, robot, `main.py`, Pi, scripts, tests — in that order.
3. Run the tests and the mock-mode smoke commands; iterate until green.
4. Update `README.md` and the relevant `docs/*.md` (setup, architecture, API, troubleshooting) so they match the final code exactly — file names, flags, endpoints, config keys.
5. Write `CLAUDE.md` (≤ 60 lines): what the project is, how to run in mock mode, where the LiteRT unknowns live, the hard design rules from §3.
6. Finish with a summary: what was built, what passed, and an explicit list of things that can only be verified on real hardware (LiteRT kwargs, Kortex wheel version, tag positions, grasp heights, `TOOL_ORIENT`).

Do not invent APIs silently. Where an API detail is uncertain, isolate it in an adapter, log clearly, and leave a `# VERIFY ON SITE:` comment explaining exactly what to check.
