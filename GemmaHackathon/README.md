# Show Me Once

**A robot arm that learns a sorting chore from one demonstration, says the rule it learned, and then does it. Everything runs on-device with Google Gemma.**

Google Gemma Hardware Hackathon · AGI House · Oct 24, 2026 · Track: Physical AI, IoT & Robotics (also: Best Use of LiteRT, Best Use of Raspberry Pi)

1. **Teach:** say "watch me", sort a few objects into bin A and bin B, say "done".
2. **Explain:** Gemma works out the rule and says it: *"Food goes in A, tools go in B."*
3. **Do:** dump new objects on the table and say "go". The arm sorts them and narrates.
4. **Correct:** say "actually, red things go in B". Gemma rewrites the rule.
5. **Offline:** pull the network. Nothing changes.

It runs as a small **swarm** of on-device nodes, each with one job: the **Pi** is the ears and mouth, the **Jetson Orin Nano Super** is the eyes, the **laptop** is the brain and hands, and the **Kinova** does the work.

Everything below is in the order you do it.

## Status and to-do (as of Oct 23)

**Done and tested on the dev laptop (Windows, RTX 4070):** LiteRT-LM Gemma E4B on the GPU (teach ≈ 21 s, plan ≈ 3 s/step); the full intent flow with a mock arm and mock camera; AprilTag detection on the printable PDF; stop, workspace and bin filtering on a synthetic scene. The `.venv` has every package, including Kortex 2.8.0, and the model is in `data/`.

**Not tested yet (needs hardware):** the real arm, the wrist camera stream, calibration, the Pi voice path, and the Jetson (its code path is tested end-to-end on a Mac stand-in with a synthetic tag scene, but not on the board).

**Tonight:**
- [ ] Print `tags/apriltags.pdf` at 100% and check the 5 cm bar (§1)
- [ ] Flash the Pi with 64-bit Raspberry Pi OS, copy `pi/` over, run `bash setup_pi.sh` (§3)
- [ ] Pi: `python voice_agent.py --text` works with the laptop server running (§7)
- [ ] Flash the Jetson (JetPack 6.x), clone the repo, run `bash jetson/setup_jetson.sh`, check `ollama list` shows `EYES_GEMMA_MODEL` (§3)
- [ ] Pack the hardware list (§1): Gigabit switch, Ethernet ×4 + USB-Ethernet adapter, mic, speaker, bins, objects, tape measure

**On site, in order:**
- [ ] Network: arm, laptop, Pi, Jetson on one switch, all ping each other (§4)
- [ ] Camera stream plays; `python -m scripts.test_gemma` (§5)
- [ ] Jetson eyes server up; laptop `/status` shows `"eyes": "jetson (online)"` (§7)
- [ ] Tape the tags, measure `TABLE_TAGS`, find `LOOK_POSE`, reach CALIBRATED (§6)
- [ ] `test_click_pick`: tune `Z_GRASP`, `TOOL_ORIENT`, `GRIPPER_CLOSED` (§6)
- [ ] Full voice loop; record a 60 s backup video by 7 PM; rehearse ×3 (§7–8)

---

## 0. How it works

```
 Raspberry Pi 5 · ears + mouth                 Laptop (RTX 4070) · brain + hands
 USB mic ─► Gemma 4 E2B (LiteRT-LM, CPU)       FastAPI :8000 (main.py)
            audio → {"intent": "go"} ─POST────►   ├─ arm.look()  (Kortex) ──────────────► Kinova Gen3
 speaker ◄─ espeak-ng ◄─── GET /say ───────────   ├─ POST /scene ─┐                         │ wrist camera
                                                  │               ▼                         │ RTSP
                                                  │   Jetson Orin Nano Super · eyes  :8001 ◄┘
                                                  │     AprilTags → pixel→metre homography
                                                  │     diff vs empty table → numbered marks
                                                  │     Gemma 4 E2B (Ollama, CUDA) names each mark
                                                  │   ◄─ {objects: id → (x, y), bins, labels, marked image}
                                                  ├─ Gemma 4 E4B (LiteRT-LM) picks "object 3 → bin A"
                                                  └─ Kortex pick_and_place(x, y) ─────────► Kinova Gen3
```

All four boxes share one Ethernet switch (§4). The Jetson reads the camera straight from the arm, so camera frames never pass through the laptop. If the Jetson fails, set `EYES = "local"` and the laptop runs the same perception code itself.

Design rules:
- **Gemma never outputs coordinates.** It picks a numbered mark and a bin letter, and deterministic code turns that into metres. VLMs point imprecisely but choose well from labelled options. The Jetson's object names are only a hint in the planner prompt; the ids and metres come from geometry.
- **Every Gemma call returns one JSON object.** The code extracts the first `{...}`, retries up to 2 times, and rejects any object id that wasn't drawn.
- **The camera is on the wrist (eye-in-hand).** Perception only uses frames taken at `LOOK_POSE`, so `empty`, `teach` and each sort step call `arm.look()` first. The homography is recomputed from the tags in every frame.
- **Safety:** every move is speed-limited (`MAX_LINEAR_SPEED`) and checked against `WORKSPACE`. `stop` aborts the current motion and the rest of the pick, and the arm stays where it is. Someone keeps a hand on the physical E-stop anyway.

### Files

```
config.py              every setting. Edit this file (no env overrides)
main.py                laptop FastAPI app: intents, state, sort loop, node health in /status
perception/camera.py   Kinova Vision RTSP reader + FakeCamera
perception/calibrate.py  AprilTag homography; run it directly to check calibration
perception/objects.py  empty-table diff, numbered marks, teach keyframes
perception/ball.py     white-ball / blue-box colour detectors (ball → box task)
perception/eyes.py     the eyes interface: LocalEyes (runs the above) and RemoteEyes (HTTP to the Jetson)
jetson/eyes_server.py  Jetson FastAPI app :8001 serving LocalEyes + object labels from its Gemma
jetson/setup_jetson.sh one-shot Jetson install (uv venv, Ollama, Gemma pull)
planner/brain.py       prompts + teach / plan / correct + rule saved to data/rule.json
planner/gemma_vision.py  LiteRT-LM and Ollama backends, JSON parsing
robot/arm.py           KinovaArm (Kortex), MockArm, pick_and_place
pi/voice_agent.py      Pi voice front end (Gemma E2B, push-to-talk, espeak)
pi/setup_pi.sh         one-shot Pi install
scripts/test_gemma.py  one camera frame → Gemma → JSON + latency
scripts/test_click_pick.py  click a pixel → arm picks there
scripts/ball_to_box.py white ball → blue box, no Gemma (runs on the arm or in sim/)
sim/                   MuJoCo sim of the cell: Gen3 + 2F-85, tags, ball → box, and sorting with Gemma (see §10)
tags/apriltags.pdf     print-ready AprilTags (tags/make_tags.py regenerates it)
```

---

## 1. Shopping and printing list

| Item | Notes |
|---|---|
| Kinova **Gen3** with the **Vision module** | The Gen3 lite has no camera |
| Laptop with NVIDIA GPU (≥ 8 GB) | Windows or Linux |
| Raspberry Pi 5 (8 GB), 27 W PSU, cooler, SD card | Raspberry Pi OS **64-bit** |
| Jetson Orin Nano Super dev kit, PSU, NVMe or SD card | JetPack **6.x** |
| Unmanaged Gigabit Ethernet switch (5+ ports) | Arm, laptop, Pi and Jetson all plug into it |
| USB mic, small speaker, optional push button (GPIO17 → GND) | |
| 4 Ethernet cables + USB-Ethernet adapter | Arm, laptop, Pi, Jetson → switch |
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

All six tags must fit in the wrist camera's view at once, so keep the tag rectangle compact, about 30 × 50 cm (the default `TABLE_TAGS`). In the sim a 35 cm deep rectangle pushed a tag off the frame.

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
If it's already on this machine, copy the existing `gemma-4-E4B-it.litertlm` into `data/` instead. Setting `HF_TOKEN` speeds up downloads.

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

## 3. Pi and Jetson install (night before, needs internet)

### Pi

```bash
cd ~/GemmaHackathon/pi
bash setup_pi.sh      # apt deps, uv venv, litert-lm, downloads Gemma 4 E2B to models/, smoke test
```
Check audio with `arecord -l`, `aplay -l` and `espeak-ng "hello"`. Set the default devices in `~/.asoundrc`.

Static IP on the swarm switch:
```bash
sudo nmcli con add type ethernet ifname eth0 con-name hack ipv4.method manual ipv4.addresses 192.168.1.12/24
sudo nmcli con up hack
```

### Jetson Orin Nano Super
```bash
git clone https://github.com/kevinhaoaus/gemma-hackathon.git && cd gemma-hackathon/GemmaHackathon
bash jetson/setup_jetson.sh   # uv venv (Python 3.11), OpenCV + AprilTags + FastAPI, Ollama, pulls EYES_GEMMA_MODEL, smoke test
sudo nmcli con add type ethernet ifname eth0 con-name hack ipv4.method manual ipv4.addresses 192.168.1.13/24
sudo nmcli con up hack
```
- Set the power mode to **MAXN SUPER**: `sudo nvpmodel -q` lists modes, then `sudo nvpmodel -m <id>`. Run `sudo jetson_clocks` before the demo.
- The Jetson runs Gemma through **Ollama** because its installer sets up CUDA on JetPack. Check the tag with `ollama list` and put it in `EYES_GEMMA_MODEL`. E2B fits easily in the 8 GB shared memory; E4B also fits but leaves less headroom.
- Labelling adds one Jetson Gemma call per sort step. If steps are too slow, set `EYES_LABELS = False`: the Jetson still does all the geometry, and the laptop planner works from the image alone.
- Keep `config.py` identical on the laptop and the Jetson (both read it). `git pull` on both after changing it.

---

## 4. Network (on site)

Everything plugs into one switch on `192.168.1.0/24`. No router and no internet are needed.

| Node | IP | Role | Listens on |
|---|---|---|---|
| Kinova Gen3 | `192.168.1.10` (default; admin / admin) | arm + wrist camera | Kortex 10000, RTSP `/color`, Web App |
| Laptop | static `192.168.1.11/24` on the switch adapter | brain + hands (`main.py`) | TCP **8000** |
| Raspberry Pi 5 | static `192.168.1.12/24` | ears + mouth (`voice_agent.py`) | — |
| Jetson Orin Nano Super | static `192.168.1.13/24` | eyes (`jetson/eyes_server.py`) | TCP **8001** |

Allow inbound TCP **8000** in the laptop firewall (and 8001 on the Jetson if `ufw` is on). Turn off the laptop's Wi-Fi if the venue network also uses `192.168.1.x`. Check: every node can `ping 192.168.1.10`; open `http://192.168.1.10` (Kinova Web App); from the Pi run `curl http://192.168.1.11:8000/status`; from the laptop run `curl http://192.168.1.13:8001/status`.

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

`test_gemma`, `perception.calibrate` and `test_click_pick` open the camera on the laptop directly, not through the Jetson, so run them from the laptop. If the stream refuses a second viewer while the Jetson server is running, stop the Jetson server first.

---

## 6. Calibration (on site, in order)

1. **Tape the tags** (§1) flat, out of the arm's path.
2. **Measure tag centres** in the robot base frame (x forward, y left, metres) and enter them in `TABLE_TAGS`.
   - Fast (about 1 cm): tape measure from the base centre.
   - Accurate (about 2 mm): in the Web App, jog the gripper tip onto each tag centre and read x, y. Also record the table height for `Z_TABLE_OFFSET`.
3. **Set `USE_MOCK_ARM = False`.** Check the tool frame in the Web App (Configurations → Robot → Tool): every height in `config.py` assumes z = 0.12 m (`SIM_TOOL_FRAME_Z`). Home should read about x 0.576, z 0.434, θ (90, 0, 90).
4. **Find `LOOK_POSE`.** In the Web App, point the gripper straight down (`TOOL_ORIENT`) high over the table centre until all six tags are in view. Copy the tool x, y, z into `LOOK_POSE`, with z measured above the table. The default `(0.40, 0.0, 0.50)` comes from the sim (§10): the old `(0.475, 0.0, 0.55)` is out of reach pointing straight down.
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
   | `Z_GRASP` | 0.062 m | misses → lower in 5 mm steps; hits the table → raise. 0.062 is the sim value for a 4 cm ball; 0.025 jams the pads into the table |
   | `TOOL_ORIENT` | (180, 0, 90) | gripper not pointing straight down → read it from the Web App |
   | `GRIPPER_CLOSED` | 0.7 | drops objects → raise; crushes them → lower |
   | `BIN_DROP` | ±0.40 m y | used only when the bin tags aren't visible |
   | `BIN_RADIUS_M` | 0.12 m | objects in a bin get picked again → raise; objects next to a bin get ignored → lower |
   | `WORKSPACE` | x 0.20–0.75, y ±0.50, z 0–0.70 m | any target outside it is refused. It must contain the tags, both bins and `LOOK_POSE` |
   | `MAX_LINEAR_SPEED` | 0.10 m/s | keep it slow near people |

   Only objects **inside the rectangle of tags 0–3** and more than `BIN_RADIUS_M` from a bin are offered to Gemma. Put the bins **outside** that rectangle.

---

## 7. Run it

Start them in this order:
```bash
# jetson (eyes) — warms its Gemma on startup
.venv/bin/uvicorn jetson.eyes_server:app --host 0.0.0.0 --port 8001
# laptop (brain + hands)
uvicorn main:app --host 0.0.0.0 --port 8000
# pi (ears + mouth)
python voice_agent.py                         # ENTER = push-to-talk (defaults to --laptop 192.168.1.11)
python voice_agent.py --text                  # type commands (debug)
python voice_agent.py --stt vosk              # if E2B audio is too slow (needs a Vosk model in pi/models/)
```
Check `curl localhost:8000/status`: `nodes` shows whether the Pi is polling, whether the eyes are online, and which arm is in use.
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
| `GET /status` | `{"mode", "rule", "calibrated", "nodes": {"pi", "eyes", "arm"}}` |

Jetson eyes API (`:8001`, called by the laptop): `POST /empty`, `POST /scene {"label"}` → objects in metres, bins, labels and the marked image (base64 JPEG), `POST /teach/start`, `POST /teach/stop` → keyframes, `GET /status`.

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

**90 s live demo:** hook (*"Robots need a programmer. Ours needs one demonstration."*) → teach 4 objects → Pi speaks the rule → a judge dumps new objects, "go" → "actually, red things go in B" → unplug the internet uplink: *"Three Gemmas on a Pi, a Jetson and a laptop. Zero cloud."*

**Judge Q&A**
- *Why not have Gemma output coordinates?* VLM pointing is imprecise. Set-of-marks plus a homography is accurate and safe.
- *What runs where?* The Pi (ears + mouth) runs E2B voice on LiteRT-LM (CPU). The Jetson (eyes) reads the wrist camera, does AprilTags and object detection, and names objects with E2B on its GPU. The laptop (brain + hands) runs E4B on LiteRT-LM (GPU text, CPU vision) at about 3 s per plan step, and drives the arm.
- *What if it's wrong?* A spoken correction rewrites the rule, invalid ids are rejected, and stop is instant.
- *Does it generalise?* Yes. The rule is abstract ("food vs tools"), so unseen objects sort correctly.

**If something breaks:** Pi down → use `/docs` or curl. Jetson down → `EYES = "local"` and restart the laptop server (retake "table is empty"). LiteRT fails → `GEMMA_BACKEND = "ollama"`. Arm faults → `USE_MOCK_ARM = True` plus the backup video.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `No frames from Kinova Vision` | Arm on? Laptop on 192.168.1.x? Does the stream play in VLC? |
| NOT CALIBRATED | All of tags 0–3 must be in view from `LOOK_POSE`: raise z, improve lighting, check the printed size |
| Shadows detected as objects | Diffuse the lighting, retake `empty`, raise the threshold in `ObjectFinder.find` |
| Two touching objects become one | Space the objects out |
| Pick is offset | Re-measure `TABLE_TAGS`. Tall objects shift outward, so keep the look pose high and vertical |
| Arm says "That's outside my safe area" | A target fell outside `WORKSPACE`. Check `TABLE_TAGS` / bin positions, then widen `WORKSPACE` only if the table really extends there |
| An object on the table is ignored | It's outside the tag 0–3 rectangle or within `BIN_RADIUS_M` of a bin |
| `kortex_api` import error | `pip install "protobuf==3.20.*"` |
| Arm action aborts | Target unreachable: move the table or bins closer, check `TOOL_ORIENT`, clear faults in the Web App |
| Gripper doesn't close | Robotiq 2F-85 uses 0–1. Tune `GRIPPER_CLOSED` |
| LiteRT `Failed to create conversation` | Wrong model file: use `gemma-4-E4B-it.litertlm`, not `-gpu` / `-web` |
| LiteRT `TypeError` on `Engine(...)` | API changed: run `help(litert_lm.Engine)` and fix the kwargs in `gemma_vision.py` / `voice_agent.py` |
| Gemma slow (> 5 s / step) | `PROMPT_IMAGE_MAX = 512`, check the GPU is in use (`nvidia-smi`) |
| Non-JSON from Gemma | Retries are automatic. Shorten the prompt in `planner/brain.py` |
| Pi can't reach laptop | Ping, firewall port 8000, server started with `--host 0.0.0.0` |
| `pip install litert-lm-api` fails on Pi | Needs a 64-bit OS (`uname -m` → `aarch64`) |
| Arm says "I can't reach the Jetson." | `curl http://192.168.1.13:8001/status` from the laptop; is the eyes server running with `--host 0.0.0.0`? Fallback: `EYES = "local"` |
| Jetson: `No frames from Kinova Vision` | The Jetson must be on the switch with the arm (`ping 192.168.1.10` from the Jetson) |
| Jetson labels missing / `[eyes] labelling failed` | `ollama list` on the Jetson must show `EYES_GEMMA_MODEL`; `systemctl status ollama`. Sorting still works without labels |
| Sort steps slow with the Jetson | `sudo nvpmodel` MAXN SUPER + `sudo jetson_clocks`, or `EYES_LABELS = False` |

| Ball task says "can't see all four table tags" | A tag is at the frame edge or behind the fingers: look at `data/last_ball.jpg`, move that tag 2 cm inwards and re-measure |

**Using a fixed USB webcam instead:** write a small OpenCV reader (`cv2.VideoCapture(0)`) as `perception/webcam.py` with the same `read()` interface, make `make_camera()` return it, and make `arm.look()` a no-op.

---

## 10. Simulation: white ball → blue box (MuJoCo)

`scripts/ball_to_box.py` finds a white ball on the black table, picks it and drops it in the blue box. It uses the same `TableMap`, `pick_and_place` and config as the real arm. `sim/` runs that exact code against a MuJoCo Gen3 + Robotiq 2F-85 (models from MuJoCo Menagerie) with a rendered wrist camera, tags, a ball and a box. `SimArm` and `SimCamera` copy the Kortex conventions (base frame, 0.12 m tool frame, θ as extrinsic XYZ, straight-line `reach_pose` at `MAX_LINEAR_SPEED`). An unreachable pose raises, like a Kortex abort. Contacts between the arm and the table, box or ball are logged per step.

```bash
uv venv sim/.venv --python 3.12
uv pip install --python sim/.venv/bin/python -r requirements-sim.txt
bash sim/get_assets.sh                                   # Gen3 + 2F-85 models → sim/menagerie
sim/.venv/bin/python -m sim.run                          # one run → sim/out/episode.mp4
sim/.venv/bin/python -m sim.run --trials 30              # randomised ball/box/tag error/lighting, success rate
sim/.venv/bin/python -m sim.run --trials 20 --sweep Z_GRASP=0.055,0.062,0.07   # tune a setting
sim/.venv/bin/python -m sim.run --trial 7                # replay randomised trial 7 with video
```
Any config value can be overridden with `--set NAME=value`. Success is ground truth (the ball is in the box), not what the script reports. Sim settings are the `SIM_*` values in `config.py`. Measure the real ball and box and put them there.

**What the sim changed** (with the old values the ball went in 0 out of 8 times):

| Setting | Was | Now | Why |
|---|---|---|---|
| `LOOK_POSE` | (0.475, 0, 0.55) | (0.40, 0, 0.50) | out of reach pointing down (wrist fully stretched); the new pose sees all 4 tags |
| `TABLE_TAGS` x | 0.30 / 0.65 | 0.32 / 0.62 | far tag off the top of the frame; near tags behind the fingers |
| `Z_GRASP` | 0.025 | 0.062 | the pads are ~4 cm below the tool frame and drop ~1 cm as they close; 100% from 0.058 to 0.065 |
| `KinovaArm._run` | abort counted as success | abort raises | an unreachable move used to carry on to the next step |

Now 30/30 randomised trials. Not modelled: a real ping-pong ball bounces a lot more than the sim's, so put a cloth in the box. The box at `BIN_DROP` A runs off the frame, so the script drops at `BIN_DROP` and confirms with a blue-ring check around the ball.

### Sorting with Gemma (`sim/sort.py`)

The full "Show Me Once" loop with the real Gemma planner (`planner/brain.py`, LiteRT-LM, `data/gemma-4-E4B-it.litertlm`) and the real `LocalEyes`, `pick_and_place` and sort loop from `main.py`:

1. Empty-table snapshot at `LOOK_POSE`.
2. **Teach:** a simulated person puts 4 demo objects into the bins by a hidden rule. Colour and shape both vary, so only one rule fits. The settled wrist-camera frames, with BIN A / BIN B drawn on them, go to `brain.teach`.
3. 6 new objects (red or blue, 4 cm balls or blocks) land at random spots on the table.
4. **Sort:** numbered marks → `brain.plan` picks the object and the bin → `pick_and_place`, until the table is clear.
5. With `--correct`: `brain.correct("actually, red things go in B")`, then a second batch.

Gemma's choices are scored against the hidden rule separately from where the objects end up, so an arm fumble isn't blamed on Gemma.

```bash
sim/.venv/bin/python -m sim.sort                         # colour rule, video → sim/out/sort.mp4, Gemma's images → sim/out/
sim/.venv/bin/python -m sim.sort --rule shape            # balls → A, blocks → B
sim/.venv/bin/python -m sim.sort --correct               # + spoken correction and a second batch
sim/.venv/bin/python -m sim.sort --trials 3 --correct    # both rules × 3 random layouts, score table
```
`--labels` adds the eyes node's labelling call (the laptop model stands in for the Jetson's E2B). Bins are `SIM_BIN_XY` / `SIM_BIN_SIZE`. On an M5 Mac: teach ≈ 25 s, plan ≈ 5 s per object, about 3 minutes per batch of 6.

Found in the sim: once a couple of objects land on a bin tag, the camera loses that tag and the code falls back to `BIN_DROP`. **Set `BIN_DROP` to the real bin centres on site**, or tape each bin tag beside the drop spot instead of under it.

---

MIT for code written at the event. Gemma models are subject to the Gemma terms of use.
