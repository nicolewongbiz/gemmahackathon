# Show Me Once

**A robot arm that learns a sorting chore from one demonstration, says the rule it learned, and then does it. Everything runs on-device with Google Gemma.**

Google Gemma Hardware Hackathon · AGI House · Oct 24, 2026 · Track: Physical AI, IoT & Robotics (also: Best Use of LiteRT, Best Use of Raspberry Pi)

1. **Teach:** say "watch me", sort a few objects into bin A and bin B, say "done".
2. **Explain:** Gemma works out the rule and says it: *"Food goes in A, tools go in B."*
3. **Do:** dump new objects on the table and say "go". The arm sorts them and narrates.
4. **Correct:** say "actually, red things go in B". Gemma rewrites the rule.
5. **Offline:** pull the network. Nothing changes.

Everything below is in the order you do it.

---

## 0. How it works

```
 Raspberry Pi 5                                Laptop (RTX 4070)
 USB mic ─► Gemma 4 E2B (LiteRT-LM, CPU)       FastAPI :8000 (main.py)
            audio → {"intent": "go"} ─POST────►   │
 speaker ◄─ espeak-ng ◄─── GET /say ───────────   ├─ arm.look() → Kinova wrist camera (RTSP) → frame
                                                  ├─ AprilTags → pixel→metre homography
                                                  ├─ diff vs empty table → numbered marks on objects
                                                  ├─ Gemma 4 E4B (LiteRT-LM GPU / Ollama) picks "object 3 → bin A"
                                                  └─ Kortex pick_and_place(x, y) ─► Kinova Gen3
```

Design rules:
- **Gemma never outputs coordinates.** It picks a numbered mark and a bin letter, and deterministic code turns that into metres. VLMs point imprecisely but choose well from labelled options.
- **Every Gemma call returns one JSON object.** The code extracts the first `{...}`, retries up to 2 times, and rejects any object id that wasn't drawn.
- **The camera is on the wrist (eye-in-hand).** Perception only uses frames taken at `LOOK_POSE`, so `empty`, `teach` and each sort step call `arm.look()` first. The homography is recomputed from the tags in every frame.
- **Safety:** slow moves, `stop` halts the arm immediately, and someone keeps a hand on the physical E-stop.

### Files

```
config.py              every setting. Edit this file (no env overrides)
main.py                laptop FastAPI app: intents, state, sort loop
perception/camera.py   Kinova Vision RTSP reader + FakeCamera
perception/calibrate.py  AprilTag homography; run it directly to check calibration
perception/objects.py  empty-table diff, numbered marks, teach keyframes
planner/brain.py       prompts + teach / plan / correct + rule saved to data/rule.json
planner/gemma_vision.py  LiteRT-LM and Ollama backends, JSON parsing
robot/arm.py           KinovaArm (Kortex), MockArm, pick_and_place
pi/voice_agent.py      Pi voice front end (Gemma E2B, push-to-talk, espeak)
pi/setup_pi.sh         one-shot Pi install
scripts/test_gemma.py  one camera frame → Gemma → JSON + latency
scripts/test_click_pick.py  click a pixel → arm picks there
archive/webcam/        old USB-webcam reader, disabled (see §9)
```

---

## 1. Shopping and printing list

| Item | Notes |
|---|---|
| Kinova **Gen3** with the **Vision module** | The Gen3 lite has no camera |
| Laptop with NVIDIA GPU (≥ 8 GB) | Windows or Linux |
| Raspberry Pi 5 (8 GB), 27 W PSU, cooler, SD card | Raspberry Pi OS **64-bit** |
| USB mic, small speaker, optional push button (GPIO17 → GND) | |
| 2 Ethernet cables + USB-Ethernet adapter | One link to the arm, one to the Pi |
| 2 bins, 10–12 soft objects with obvious categories | Foam fruit, sponges, small plastic tools, red/blue items, 3–7 cm wide |
| Tape, tape measure, marker, power strip | |
| **AprilTags**, family **tag36h11**, IDs **0, 1, 2, 3, 10, 11** | See below |

**Printing the tags:** get the PNGs from the AprilRobotics `apriltag-imgs` GitHub repo (folder `tag36h11/`, files `tag36_11_00000.png` and so on). Scale them so the **black square measures exactly 5.0 cm** (`TAG_SIZE_M`), and print at 100%. Print spares.

| ID | Goes on |
|---|---|
| 0 | Table corner nearest the arm, right |
| 1 | Table corner nearest the arm, left |
| 2 | Far left corner |
| 3 | Far right corner |
| 10 | Bin A |
| 11 | Bin B |

All six tags must fit in the wrist camera's view at once, so keep the tag rectangle compact, about 35 × 50 cm (the default `TABLE_TAGS`).

---

## 2. Laptop install (night before, needs internet)

```bash
cd GemmaHackathon
py -3.11 -m venv .venv            # Linux: python3.11 -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements-laptop.txt
```
Use Python **3.11**. Newer versions can lack wheels for `pupil-apriltags`, and Kortex needs the old `protobuf`.

**Kortex API** (not on PyPI): download `kortex_api-2.x.x-py3-none-any.whl` from Kinova's Artifactory (linked in the `Kinovarobotics/Kinova-kortex2_Gen3_G3L` README), then:
```bash
pip install kortex_api-*.whl
pip install "protobuf==3.20.*"     # if kortex_api fails to import
```

**Gemma 4 E4B model** → `data/gemma-4-E4B-it.litertlm` (3.7 GB):
```bash
hf download litert-community/gemma-4-E4B-it-litert-lm gemma-4-E4B-it.litertlm --local-dir data
```
If it's already on this machine, copy `../GemmaHackathonBasic/models/gemma-4-E4B-it-litert-lm/gemma-4-E4B-it.litertlm` into `data/` instead. Setting `HF_TOKEN` speeds up downloads.

> Use **`gemma-4-E4B-it.litertlm`**, not the `-gpu` or `-web` variants. Those have no vision encoder and fail with `Failed to create conversation` as soon as an image is sent.

**LiteRT-LM works on Windows** (`litert-lm-api` 0.18.1 ships a `win_amd64` wheel). Measured on the RTX 4070 laptop, with the GPU running text and the CPU running vision, as set in `gemma_vision.py`:

| Call | Time |
|---|---|
| Teach (4 frames, includes ~2 s model load) | ~21 s |
| Plan (one step) | ~3 s |
| Correct | ~2 s |

If pip says `No matching distribution found for litert-lm-api`, the pip cache is corrupt: run `pip install --no-cache-dir litert-lm-api`.

**Ollama fallback** (only if LiteRT misbehaves): run `ollama pull gemma4:e4b` and set `GEMMA_BACKEND = "ollama"`. It's slower: about 6–9 s per call, and about 45 s for the first one.

---

## 3. Pi install (night before, needs internet)

```bash
cd ~/GemmaHackathon/pi
bash setup_pi.sh      # apt deps, uv venv, litert-lm, downloads Gemma 4 E2B to models/, smoke test
```
Check audio with `arecord -l`, `aplay -l` and `espeak-ng "hello"`. Set the default devices in `~/.asoundrc`.

Static IP for the laptop link:
```bash
sudo nmcli con add type ethernet ifname eth0 con-name hack ipv4.method manual ipv4.addresses 192.168.50.2/24
sudo nmcli con up hack
```

---

## 4. Network (on site)

| Link | Laptop side | Other side |
|---|---|---|
| Laptop ↔ Kinova | static `192.168.1.11/24` on the arm adapter | arm `192.168.1.10` (admin / admin) |
| Laptop ↔ Pi | static `192.168.50.1/24` | Pi `192.168.50.2/24` |

Allow inbound TCP **8000** in the laptop firewall. Check: `ping 192.168.1.10`, open `http://192.168.1.10` (Kinova Web App), and from the Pi run `curl http://192.168.50.1:8000/status` once the server is up.

---

## 5. Camera check (Kinova Vision module)

1. In the Web App camera page you should see live video. Set the colour stream to **1280×720**, which matches `FRAME_W/FRAME_H`.
2. Optional: play `rtsp://192.168.1.10/color` in VLC. If VLC plays it, OpenCV can read it too.
3. Gemma + camera test (`USE_MOCK_ARM` doesn't matter here):
   ```bash
   python -m scripts.test_gemma            # or: python -m scripts.test_gemma ollama
   ```
   It prints the objects Gemma sees and the latency. Running it also warms the model, so do it before every demo. If calls are slow, set `PROMPT_IMAGE_MAX = 512`.

Set `USE_MOCK_CAMERA = True` to run anything without the arm (it returns grey frames).

---

## 6. Calibration (on site, in order)

1. **Tape the tags** (§1) flat, out of the arm's path.
2. **Measure tag centres** in the robot base frame (x forward, y left, metres) and enter them in `TABLE_TAGS`.
   - Fast (about 1 cm): tape measure from the base centre.
   - Accurate (about 2 mm): in the Web App, jog the gripper tip onto each tag centre and read x, y. Also record the table height for `Z_TABLE_OFFSET`.
3. **Set `USE_MOCK_ARM = False`.**
4. **Find `LOOK_POSE`.** In the Web App, point the gripper straight down (`TOOL_ORIENT`) high over the table centre until all six tags are in view. Copy the tool x, y, z into `LOOK_POSE`, with z measured above the table. The default `(0.475, 0.0, 0.55)` is a guess.
5. **Verify:**
   ```bash
   python -m perception.calibrate     # moves the arm to LOOK_POSE, shows the live view
   ```
   The status must read **CALIBRATED**. Click a tag centre and the printed metres should match `TABLE_TAGS` within 1 cm. Gripper fingers at the edge of the frame are fine.
6. **Tune grasps** (E-stop in hand):
   ```bash
   python -m scripts.test_click_pick  # click an object → arm picks it → drops in bin A
   ```
   | Setting | Start | Tune if… |
   |---|---|---|
   | `Z_HOVER` | 0.15 m | arm hits things on approach → raise |
   | `Z_GRASP` | 0.025 m | misses → lower in 5 mm steps; hits the table → raise |
   | `TOOL_ORIENT` | (180, 0, 90) | gripper not pointing straight down → read it from the Web App |
   | `GRIPPER_CLOSED` | 0.7 | drops objects → raise; crushes them → lower |
   | `BIN_DROP` | ±0.40 m y | used only when the bin tags aren't visible |

---

## 7. Run it

```bash
# laptop
uvicorn main:app --host 0.0.0.0 --port 8000
# pi
python voice_agent.py --laptop 192.168.50.1          # ENTER = push-to-talk
python voice_agent.py --text --laptop 192.168.50.1   # type commands (debug)
python voice_agent.py --stt vosk --laptop 192.168.50.1   # if E2B audio is too slow (needs a Vosk model in pi/models/)
```
For a hardware button, set `USE_GPIO = True` in `voice_agent.py` (GPIO17).

**Say, in order:** "table is empty" → "watch me" → *(sort 3–4 objects)* → "done" → *(dump new objects)* → "go" → "actually, red things go in B" → "stop".

Lighting must not change between "table is empty" and sorting, or the diff breaks. If it changes, retake the snapshot. During "watch me" the arm hovers at the look pose while you work underneath it.

**No Pi?** Drive it from `http://localhost:8000/docs` or with curl:
```bash
curl -X POST localhost:8000/intent -H "content-type: application/json" -d "{\"intent\":\"empty\"}"
# intents: empty | teach | done | go | stop | correct (with "text": "red things go in B")
```

### API

| Endpoint | Purpose |
|---|---|
| `POST /intent` `{"intent", "text"}` | Run an intent. Returns `{"ok", "mode", "rule"}` |
| `GET /say` | Drains the speech queue. The Pi polls it every 0.5 s |
| `GET /status` | `{"mode", "rule", "calibrated"}` |

The latest marked image Gemma saw is saved to `data/last_marked.jpg`.

---

## 8. Demo day

| Time | Goal |
|---|---|
| 10:00 | Claim a bench near power with room for the arm |
| 12:15 | Start building: network, camera, calibration first |
| 2:00 | **M1:** click-to-pick works |
| 4:00 | **M2:** frame → Gemma → arm picks, Pi "go" works |
| 6:00 | **M3:** teach + explain + correct |
| 7:00 | Project draft saved, 60 s backup video recorded |
| 8:00 | Demos and judging |

**90 s live demo:** hook (*"Robots need a programmer. Ours needs one demonstration."*) → teach 4 objects → Pi speaks the rule → a judge dumps new objects, "go" → "actually, red things go in B" → unplug the network: *"Two Gemma models, a Pi and a laptop. Zero cloud."*

**Judge Q&A**
- *Why not have Gemma output coordinates?* VLM pointing is imprecise. Set-of-marks plus a homography is accurate and safe.
- *What runs where?* The Pi runs E2B voice on LiteRT-LM (CPU). The laptop runs E4B on LiteRT-LM (GPU text, CPU vision) at about 3 s per plan step.
- *What if it's wrong?* A spoken correction rewrites the rule, invalid ids are rejected, and stop is instant.
- *Does it generalise?* Yes. The rule is abstract ("food vs tools"), so unseen objects sort correctly.

**If something breaks:** Pi down → use `/docs` or curl. LiteRT fails → `GEMMA_BACKEND = "ollama"`. Arm faults → `USE_MOCK_ARM = True` plus the backup video.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `No frames from Kinova Vision` | Arm on? Laptop on 192.168.1.x? Does the stream play in VLC? |
| NOT CALIBRATED | All of tags 0–3 must be in view from `LOOK_POSE`: raise z, improve lighting, check the printed size |
| Shadows detected as objects | Diffuse the lighting, retake `empty`, raise the threshold in `ObjectFinder.find` |
| Two touching objects become one | Space the objects out |
| Pick is offset | Re-measure `TABLE_TAGS`. Tall objects shift outward, so keep the look pose high and vertical |
| `kortex_api` import error | `pip install "protobuf==3.20.*"` |
| Arm action aborts | Target unreachable: move the table or bins closer, check `TOOL_ORIENT`, clear faults in the Web App |
| Gripper doesn't close | Robotiq 2F-85 uses 0–1. Tune `GRIPPER_CLOSED` |
| LiteRT `Failed to create conversation` | Wrong model file: use `gemma-4-E4B-it.litertlm`, not `-gpu` / `-web` |
| LiteRT `TypeError` on `Engine(...)` | API changed: run `help(litert_lm.Engine)` and fix the kwargs in `gemma_vision.py` / `voice_agent.py` |
| Gemma slow (> 5 s / step) | `PROMPT_IMAGE_MAX = 512`, check the GPU is in use (`nvidia-smi`) |
| Non-JSON from Gemma | Retries are automatic. Shorten the prompt in `planner/brain.py` |
| Pi can't reach laptop | Ping, firewall port 8000, server started with `--host 0.0.0.0` |
| `pip install litert-lm-api` fails on Pi | Needs a 64-bit OS (`uname -m` → `aarch64`) |

**Using a fixed USB webcam instead:** `archive/webcam/camera.py` holds the old OpenCV reader. Copy it to `perception/webcam.py`, make `make_camera()` return `webcam.Camera(0)`, and make `arm.look()` a no-op.

---

MIT for code written at the event. Gemma models are subject to the Gemma terms of use.
