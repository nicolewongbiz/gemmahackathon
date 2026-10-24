# Show Me Once

**A robot arm that learns a sorting chore from one demonstration, says the rule it learned, and then does it. Everything runs on-device with Google Gemma.**

Google Gemma Hardware Hackathon · AGI House · Oct 24, 2026 · Track: Physical AI, IoT & Robotics (also: Best Use of LiteRT, Best Use of Raspberry Pi)

1. **Teach:** say "watch me", sort a few objects into bin A and bin B, say "done".
2. **Explain:** Gemma works out the rule and says it: *"Food goes in A, tools go in B."*
3. **Do:** dump new objects on the table and say "go". The arm sorts them and narrates.
4. **Correct:** say "actually, red things go in B". Gemma rewrites the rule.
5. **Offline:** pull the network. Nothing changes.

Everything below is in the order you do it.

## Status and to-do (as of Oct 23)

**Done and tested on the dev laptop (Windows, RTX 4070):** LiteRT-LM Gemma E4B on the GPU (teach ≈ 21 s, plan ≈ 3 s/step); the full intent flow with a mock arm and mock camera; AprilTag detection on the printable PDF; stop, workspace and bin filtering on a synthetic scene. The `.venv` has every package, including Kortex 2.8.0, and the model is in `data/`.

**Not tested yet (needs hardware):** picking with the real arm (it connects), the USB camera on its stand, calibration, and the Pi voice path. Our arm has **no Kinova Vision module**, so the project uses a USB webcam (`CAMERA = "webcam"`).

**Tonight:**
- [ ] Print `tags/apriltags.pdf` at 100% and check the 5 cm bar (§1)
- [ ] Flash the Pi with 64-bit Raspberry Pi OS, copy `pi/` over, run `bash setup_pi.sh` (§3)
- [ ] Pi: `python voice_agent.py --text` works with the laptop server running (§7)
- [ ] Pack the hardware list (§1): USB webcam + stand/clamp, Ethernet ×2 + USB-Ethernet adapter, mic, speaker, bins, objects, tape measure

**On site, in order:**
- [ ] Network: laptop ↔ arm, laptop ↔ Pi (§4)
- [ ] Mount the webcam above the table; `python -m scripts.test_gemma` (§5)
- [ ] Tape the tags, measure `TABLE_TAGS`, find `LOOK_POSE`, reach CALIBRATED (§6)
- [ ] `measure_tags.bat`, then `click_pick.bat`: tune `Z_GRASP`, `GRIPPER_CLOSE_SPEED` (§6)
- [ ] Full voice loop; record a 60 s backup video by 7 PM; rehearse ×3 (§7–8)

---

## 0. How it works

```
 Raspberry Pi 5                                Laptop (RTX 4070)
 USB mic ─► Gemma 4 E2B (LiteRT-LM, CPU)       FastAPI :8000 (main.py)
            audio → {"intent": "go"} ─POST────►   │
 speaker ◄─ espeak-ng ◄─── GET /say ───────────   ├─ arm.look() (out of view) → USB webcam → frame
                                                  ├─ AprilTags → pixel→metre homography
                                                  ├─ diff vs empty table → numbered marks on objects
                                                  ├─ Gemma 4 E4B (LiteRT-LM GPU / Ollama) picks "object 3 → bin A"
                                                  └─ Kortex pick_and_place(x, y) ─► Kinova Gen3
```

Design rules:
- **Gemma never outputs coordinates.** It picks a numbered mark and a bin letter, and deterministic code turns that into metres. VLMs point imprecisely but choose well from labelled options.
- **Every Gemma call returns one JSON object.** The code extracts the first `{...}`, retries up to 2 times, and rejects any object id that wasn't drawn.
- **Camera: a USB webcam on a stand above the table** (`CAMERA = "webcam"`, `CAMERA_ON_WRIST = False`). Before every perception frame (`empty`, `teach`, each sort step) the code calls `arm.look()`, which parks the arm at `LOOK_POSE` so it isn't in the picture. With a wrist-mounted camera (`CAMERA_ON_WRIST = True`, or `CAMERA = "kinova"` on a Gen3 with the Vision module) `LOOK_POSE` is the viewing pose instead. The homography is recomputed from the tags in every frame.
- **Safety:** every move is speed-limited (`MAX_LINEAR_SPEED`) and checked against `WORKSPACE`. `stop` aborts the current motion and the rest of the pick, and the arm stays where it is. Someone keeps a hand on the physical E-stop anyway.

### Files

```
config.py              every setting. Edit this file (no env overrides)
*.bat                  double-click launchers (use the .venv, start in this folder, keep the window open):
                       run, calibrate, measure_tags, click_pick, find_camera, test_gemma
main.py                laptop FastAPI app: intents, state, sort loop
perception/camera.py   USB webcam, Kinova Vision RTSP reader, FakeCamera (config.CAMERA picks)
perception/calibrate.py  AprilTag homography; run it directly to check calibration
perception/objects.py  empty-table diff, numbered marks, teach keyframes
planner/brain.py       prompts + teach / plan / correct + rule saved to data/rule.json
planner/gemma_vision.py  LiteRT-LM and Ollama backends, JSON parsing
robot/arm.py           KinovaArm (Kortex), MockArm, pick_and_place
pi/voice_agent.py      Pi voice front end (Gemma E2B, push-to-talk, espeak)
pi/setup_pi.sh         one-shot Pi install
scripts/test_gemma.py  one camera frame → Gemma → JSON + latency
scripts/test_click_pick.py  click a pixel → arm picks there
scripts/find_camera.py  saves a frame per camera index, to find WEBCAM_INDEX
scripts/measure_tags.py  reads the arm pose at each tag and writes TABLE_TAGS
tags/apriltags.pdf     print-ready AprilTags (tags/make_tags.py regenerates it)
```

---

## 1. Shopping and printing list

| Item | Notes |
|---|---|
| Kinova **Gen3** | Ours has no Vision module, so we use a USB webcam |
| USB webcam (1080p) + a stand or clamp | Mounted ~80–100 cm above the table, looking straight down. The laptop camera works for a desk test |
| Laptop with NVIDIA GPU (≥ 8 GB) | Windows or Linux |
| Raspberry Pi 5 (8 GB), 27 W PSU, cooler, SD card | Raspberry Pi OS **64-bit** |
| USB mic, small speaker, optional push button (GPIO17 → GND) | |
| 2 Ethernet cables + USB-Ethernet adapter | One link to the arm, one to the Pi |
| 2 bins, 10–12 soft objects with obvious categories | Foam fruit, sponges, small plastic tools, red/blue items, 3–7 cm wide |
| Tape, tape measure, marker, power strip | |
| **AprilTags**, family **tag36h11**, IDs **0, 1, 2, 3, 10, 11** | See below |

**Printing the tags:** print **`tags/apriltags.pdf`** on US Letter at **100% / "Actual size"**, not "Fit to page". Page 1 has all six labelled tags and page 2 has spares. Check the 5 cm bar at the top with a ruler: the black squares must measure 5.0 cm (`TAG_SIZE_M`). Cut along the grey lines and leave the white border on. To regenerate the PDF after changing the IDs or the size in `config.py`, run `python tags/make_tags.py`.

| ID | Goes on |
|---|---|
| 0 | Table corner nearest the arm, right |
| 1 | Table corner nearest the arm, left |
| 2 | Far left corner |
| 3 | Far right corner |
| 10 | Bin A |
| 11 | Bin B |

All six tags must fit in the camera's view at once, so keep the tag rectangle compact, about 35 × 50 cm (the default `TABLE_TAGS`).

---

## 2. Laptop install (night before, needs internet)

```bash
cd GemmaHackathon
py -3.11 -m venv .venv            # Linux: python3.11 -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements-laptop.txt
```
Use Python **3.11**. Newer versions can lack wheels for `pupil-apriltags`, and Kortex needs the old `protobuf`.

**Kortex API** (not on PyPI, but a public download with no login):
```bash
curl -LO https://artifactory.kinovaapps.com/artifactory/generic-local-public/kortex/API/2.8.0/kortex_api-2.8.0.post5-py3-none-any.whl
pip install kortex_api-2.8.0.post5-py3-none-any.whl     # pulls protobuf 3.20, which is what it needs
```
Older versions (2.6, 2.7) live next to it on Kinova's Artifactory if the arm's firmware is old. `*.whl` is gitignored.

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

**Double-click the `.bat` files, not the `.py` files.** Double-clicking a `.py` runs the system Python without the `.venv` packages, so it crashes and the window closes instantly. Each `.bat` uses `.venv`, runs from the project folder and pauses at the end so you can read errors. From a terminal, the `python -m ...` commands in this guide work after `.venv\Scripts\activate`.

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

## 5. Camera check (USB webcam)

1. **Mount the webcam** on a stand or clamp about 80–100 cm above the table centre, pointing straight down, beside the arm rather than over its base so the arm rarely blocks it. Use a USB-A port on the laptop.
2. **Pick the camera.** Double-click `find_camera.bat`: it saves a frame from every camera index to `data/cam<i>.jpg`. Open them and set `WEBCAM_INDEX` to the webcam's (on the dev laptop the Brio 100 is `0`, the built-in camera `1` and OBS Virtual Camera `2`, but the order can change when devices are plugged in). Close Teams, Zoom, OBS and the Windows Camera app, since only one program can use a camera at a time.
3. Gemma + camera test (`USE_MOCK_ARM` doesn't matter here):
   ```bash
   test_gemma.bat                     # = python -m scripts.test_gemma; add ollama for the fallback backend
   ```
   It prints the objects Gemma sees and the latency. Running it also warms the model, so do it before every demo. If calls are slow, set `PROMPT_IMAGE_MAX = 512`.

Set `CAMERA = "mock"` to run anything without a camera (it returns grey frames). `CAMERA = "kinova"` reads a Gen3 Vision module at `rtsp://<ROBOT_IP>/color`, if you ever use an arm that has one.

---

## 6. Calibration (on site, in order)

1. **Tape the tags** (§1) flat, out of the arm's path.
2. **Measure tag centres** in the robot base frame. This is what makes picks accurate: every pick is mapped through these four points, so a 2 cm error here is a 2 cm miss.
   - Best (about 2 mm): double-click `measure_tags.bat`. For each tag, jog the gripper (pointing down) in the Web App until the fingertip centre is over the tag centre, then press Enter. It reads the arm's pose and writes `TABLE_TAGS` into `config.py`. It never moves the arm.
   - Quick (about 1–2 cm): tape measure from the base centre (x forward, y left, metres).
3. **Set `USE_MOCK_ARM = False`.**
4. **Find `LOOK_POSE`**, the pose where the arm waits while the camera takes a frame. In the Web App, jog the arm to the side so it covers none of the six tags in the camera view, and copy the tool x, y, z into `LOOK_POSE` (z above the table). It must be inside `WORKSPACE`. On the dev table it is the Kinova Home pose, `(0.579, -0.003, 0.433)` with the gripper pointing forward (`LOOK_ORIENT = (90.4, 0.0, 89.7)`); copy both from the Web App's End Effector panel (X, Y, Z, Theta x/y/z). With a wrist camera it is instead the pose that looks down on all six tags.
5. **Verify:**
   ```bash
   calibrate.bat                      # = python -m perception.calibrate; parks the arm at LOOK_POSE, shows the live view
   ```
   The status must read **CALIBRATED**. Click a tag centre and the printed metres should match `TABLE_TAGS` within 1 cm. Gripper fingers at the edge of the frame are fine.
6. **Tune grasps** (E-stop in hand):
   ```bash
   click_pick.bat                     # = python -m scripts.test_click_pick; click an object → arm picks it → drops in bin A
   ```
   Press **E** with the table clear to save it empty. Objects then get a yellow outline and cross, and a click anywhere on an object picks at its detected centre, so you don't have to click precisely.

   | Setting | Start | Tune if… |
   |---|---|---|
   | `Z_HOVER` | 0.25 m | arm hits things on approach → raise |
   | `Z_GRASP` | 0.135 m (dev table) | jog the open gripper around an object in the Web App and copy End Effector Z; then fine-tune in 5 mm steps |
   | `TOOL_ORIENT` | (180, 0, 90) | gripper not pointing straight down → read it from the Web App |
   | `GRIPPER_CLOSE_SPEED` | 0.3 | the gripper closes until it stalls on the object, so any size works. Crushes soft objects → lower; slips → raise |
   | `BIN_DROP` | ±0.40 m y | used only when the bin tags aren't visible |
   | `BIN_RADIUS_M` | 0.12 m | objects in a bin get picked again → raise; objects next to a bin get ignored → lower |
   | `WORKSPACE` | x 0.20–0.75, y ±0.50, z 0–0.70 m | any target outside it is refused. It must contain the tags, both bins and `LOOK_POSE` |
   | `MAX_LINEAR_SPEED` | 0.10 m/s | keep it slow near people |

   Only objects **inside the rectangle of tags 0–3** and more than `BIN_RADIUS_M` from a bin are offered to Gemma. Put the bins **outside** that rectangle.

---

## 7. Run it

```bash
# laptop
uvicorn main:app --host 0.0.0.0 --port 8000     # or double-click run.bat
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
| `No frames from webcam` | Another app has the camera, or wrong `WEBCAM_INDEX`: try 0, 1, 2 |
| Camera shows the wrong view (laptop camera, OBS logo) | `python -m scripts.find_camera`, then fix `WEBCAM_INDEX` |
| The arm is in the picture | Move `LOOK_POSE` further to the side, or move the camera |
| NOT CALIBRATED | All of tags 0–3 must be in view and not covered by the arm: raise the camera, improve lighting, check the printed size |
| Shadows detected as objects | Diffuse the lighting, retake `empty`, raise the threshold in `ObjectFinder.find` |
| Two touching objects become one | Space the objects out |
| Pick is offset | Re-measure `TABLE_TAGS`. Tall objects shift outward, so keep the camera high and pointing straight down |
| Arm says "That's outside my safe area" | A target fell outside `WORKSPACE`. Check `TABLE_TAGS` / bin positions, then widen `WORKSPACE` only if the table really extends there |
| An object on the table is ignored | It's outside the tag 0–3 rectangle or within `BIN_RADIUS_M` of a bin |
| `kortex_api` import error | `pip install "protobuf==3.20.*"` |
| Arm action aborts | Target unreachable: move the table or bins closer, check `TOOL_ORIENT`, clear faults in the Web App |
| Gripper doesn't close | The console prints `grasp: closed from X to Y`. If Y doesn't change, test the gripper in the Web App first. `(nothing in the gripper?)` means it closed on air: check `Z_GRASP` and the pick position |
| Loud jolt and a fault when the wrist turns | Big orientation changes must use `arm.move_joints` (joint space), not `move_to` (straight line). Clear faults in the Web App; the arm also clears them on connect |
| LiteRT `Failed to create conversation` | Wrong model file: use `gemma-4-E4B-it.litertlm`, not `-gpu` / `-web` |
| LiteRT `TypeError` on `Engine(...)` | API changed: run `help(litert_lm.Engine)` and fix the kwargs in `gemma_vision.py` / `voice_agent.py` |
| Gemma slow (> 5 s / step) | `PROMPT_IMAGE_MAX = 512`, check the GPU is in use (`nvidia-smi`) |
| Non-JSON from Gemma | Retries are automatic. Shorten the prompt in `planner/brain.py` |
| Pi can't reach laptop | Ping, firewall port 8000, server started with `--host 0.0.0.0` |
| `pip install litert-lm-api` fails on Pi | Needs a 64-bit OS (`uname -m` → `aarch64`) |

---

MIT for code written at the event. Gemma models are subject to the Gemma terms of use.
