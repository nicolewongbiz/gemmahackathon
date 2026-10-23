# Architecture

## Overview

```
 ┌──────────── Raspberry Pi 5 ────────────┐        ┌──────────────── Laptop (RTX 4070) ─────────────────┐
 │                                         │ intent │                                                    │
 │  USB mic ─► Gemma 4 E2B (LiteRT-LM CPU) ├───────►│  FastAPI :8000 ──go/teach──► Gemma 4 E4B (LiteRT GPU)│
 │             audio → intent JSON         │        │      │                        ▲   picks id + bin    │
 │                                         │ /say   │      │                        │                     │
 │  Speaker ◄─ espeak-ng ◄─────────────────┤◄───────┤      │          Webcam ─► Perception               │
 │                                         │        │      │                     AprilTag homography     │
 └─────────────────────────────────────────┘        │      ▼                     numbered object marks   │
                                                    │  Kortex pick_and_place ◄── pixels → metres        │
                                                    └──────┬─────────────────────────────────────────────┘
                                                           ▼
                                                      Kinova arm
```

## Components

| Component | Device | Module | Responsibility |
|---|---|---|---|
| Voice agent | Pi | `pi/voice_agent.py` | Push-to-talk, record 4 s @16 kHz, Gemma E2B → intent JSON, POST to laptop, speak `/say` |
| LiteRT adapter | Pi + laptop | `*/litert_adapter.py` | Single place that touches `litert_lm`; `chat(system, text, image_paths, audio_path)` |
| HTTP server | Laptop | `main.py` | Intents in, speech out, status, debug frame, backup web UI |
| State machine | Laptop | `main.py` | `idle → teaching → thinking → sorting → idle` |
| Camera | Laptop | `perception/camera.py` | Threaded capture, `FakeCamera` for tests |
| Calibration | Laptop | `perception/calibrate.py` | Tag detection, homography px→m, bin locations, bounds |
| Object finder | Laptop | `perception/objects.py` | Empty-table diff, centroids, numbered marks, keyframes |
| Planner | Laptop | `planner/brain.py` | teach / plan / correct; rule persisted in `data/rule.json` |
| Backends | Laptop | `planner/backends.py` | `litert` (primary), `ollama` (fallback), `fake` (tests) |
| Arm | Laptop | `robot/arm.py`, `robot/motions.py` | Kortex session, cartesian moves, gripper, stop, bounds-checked pick-and-place |

## Why two models on two devices

- **Gemma 4 E2B on the Pi**: ~9 tok/s decode, ~1.4 GB RAM on a Pi 5 (Google's figures). Fine for a 20-token intent; too slow for multi-image vision planning. Native audio input removes a speech-to-text stage.
- **Gemma 4 E4B on the laptop GPU**: multi-image reasoning for teach + per-step planning; fits in 8 GB VRAM with headroom for the webcam pipeline.
- Both via **LiteRT-LM** → one runtime story for the LiteRT prize.

## Data flow per sort step

1. `cam.read()` → frame
2. `TableMap.update(frame)` → homography (cached if the arm occludes a tag)
3. `ObjectFinder.find(frame, tag_mask)` → `{1:(u,v), 2:(u,v), …}`
4. `draw_marks()` → marked image saved as `data/last_marked.jpg`, served at `/frame.jpg`
5. `Brain.plan(marked, ids)` → `{"object": 3, "bin": "A", "label": "banana", "reason": "food"}`
6. Validate id ∈ ids, bin ∈ {A,B}
7. `px_to_table(u,v)` → (x,y) metres; `bin_xy("A")`
8. `pick_and_place()` — bounds check, stop check between motions
9. `say("banana, bin A")` → queue → Pi speaks
10. Arm returns home (out of camera view) → loop

## Teach flow

`teach` intent starts a `KeyframeRecorder` thread at 15 fps. A frame is kept each time the scene changes and then stays still for `STILL_FRAMES`. On `done`, up to `KEYFRAME_MAX` (6) frames are subsampled and sent in one Gemma call with the `TEACH` prompt.

## Design rules

1. **Gemma never outputs coordinates.** Set-of-marks only. Geometry is deterministic.
2. **JSON in, JSON out.** Robust extraction, schema validation, id whitelist, retry ×2.
3. **Every hardware dependency is mockable** (`MockArm`, `FakeCamera`, `FakeBackend`).
4. **Safety first:** bounds-checked targets, speed limits, a `stop` that interrupts any thread, physical E-stop within reach.
5. **No cloud.** Models are downloaded the night before; the demo runs on an air-gapped link.

## State machine

| State | Entered by | Exits to |
|---|---|---|
| idle | startup, end of sort, `stop` | teaching (`teach`), sorting (`go`) |
| teaching | `teach` | thinking (`done`) |
| thinking | `done`, `correct` | idle |
| sorting | `go` | idle (empty table, `stop`, error, `MAX_PICKS`) |

`empty` (snapshot the empty table) is allowed only in `idle`.
