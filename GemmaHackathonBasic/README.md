# Gemma ⇄ Kinova Gen3 (basic)

A chat box on your laptop. Talk to Gemma normally, or tell it to move the arm in plain English:
"move the hand 5 cm forward", "go up 10 cm then open the gripper", "rotate the wrist 20 degrees", "go home".
Gemma runs fully on the laptop (LiteRT-LM). There is no cloud and no Raspberry Pi.

## 1. Connect the arm (Kinova Gen3, 7-DoF)

The full Gen3 connects over **Ethernet**, not USB.

1. Run an Ethernet cable from the **RJ45 port on the arm's base** to the laptop. If the laptop has no Ethernet port, use a **USB-to-Ethernet adapter**. From the code's point of view that is still Ethernet.
2. Give the laptop a static IP of **192.168.1.11 / 255.255.255.0** on that adapter. Run PowerShell **as Administrator**:
   ```
   .\scripts\set_robot_ip_windows.ps1 -Adapter "Ethernet"
   ```
   (`Get-NetAdapter` lists the adapter names. On Linux, run `sudo bash scripts/set_robot_ip_linux.sh eth0`.)
3. The arm's default IP is **192.168.1.10**. Open http://192.168.1.10 in a browser to reach the Kinova Web App (login admin / admin). If it loads, the connection works.

The Gen3 also has a micro-USB port on its base. Ethernet is the documented and reliable path, so use that.

## 2. Install (once, needs internet)

1. Download the **Kortex API wheel** (`kortex_api-2.x.x-py3-none-any.whl`) from Kinova's Artifactory. The link is in the [Kinova-kortex2_Gen3_G3L](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L) README. Drop the file into this folder.
2. Run:
   ```
   powershell -ExecutionPolicy Bypass -File setup_windows.ps1      (Windows)
   bash setup.sh                                                    (Linux/macOS)
   ```
   The setup script:
   - creates `.venv` (Python 3.11 recommended)
   - installs the requirements and the Kortex wheel
   - downloads Gemma 4 E4B for LiteRT-LM into `models/gemma.litertlm`
   - runs a self-test

## 3. Run

| Command | What it does |
|---|---|
| `run_mock.bat` | Gemma plus a fake arm. Try this first. |
| `run_mock.bat --backend rules` | No model: a regex parser stands in for Gemma, to test the arm path |
| `run_real.bat` | Gemma plus the real Gen3. **Keep the E-stop in reach.** |
| `run_real.bat --backend ollama` | Real arm, with Gemma through Ollama (fallback) |
| `./run.sh --mock` or `./run.sh --real` | The same, on Linux/macOS |

Then open http://127.0.0.1:7860.

- The **STOP** button and the **Esc** key stop the arm immediately. They skip Gemma entirely.
- Uncheck **Execute arm actions** for a dry run, which shows what Gemma would do without moving anything.

## 4. Check scripts

| Script | Purpose |
|---|---|
| `python scripts/selftest.py` | Offline test: parser, safety limits, mock execution |
| `python scripts/check_gemma.py` | Runs Gemma on sample commands and prints its JSON and latency |
| `python scripts/check_gemma.py --probe` | Shows what the installed `litert_lm` offers. Use it if GPU fails. |
| `python scripts/check_robot.py` | Connects to the arm and prints its pose (no motion) |
| `python scripts/check_robot.py --move` | Opens and closes the gripper, then moves 3 cm up and back |
| `python scripts/download_model.py --repo litert-community/gemma-4-E2B-it-litert-lm` | Switches to the smaller, faster E2B model |

Use the venv Python: `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Linux/macOS.

## How it works

```
you type ─► app.py ─► Gemma (LiteRT-LM, on the laptop)
                         │  {"reply": "...", "actions": [{"type":"move","dx":0.05,...}]}
                         ▼
                    commands.py  validate → clamp → workspace check
                         ▼
                    arm.py  Kortex: read pose → add offset → reach_pose (speed-limited)
```

- **Directions** are in the robot base frame, as seen standing behind the arm: forward = +x, left = +y, up = +z.
- **Gemma actions:** `move` (relative), `move_to` (absolute), `rotate`, `gripper` (0 = open, 1 = closed), `home`, `stop`, `wait`.
- Each message includes the current hand position, so you can ask "where is the hand?".

## Safety limits (all in `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `LINEAR_SPEED` | 0.10 m/s | Translation speed |
| `ANGULAR_SPEED` | 20 °/s | Rotation speed |
| `MAX_STEP_M` | 0.20 m | Largest single move; larger requests are refused |
| `MAX_ROT_DEG` | 45° | Largest single rotation |
| `WORKSPACE` | x 0.15–0.85, y −0.55–0.55, z 0.03–0.80 m | The hand must stay inside this box. **Tune it to your table.** |

Any setting can be overridden with an environment variable, e.g. `set GB_LINEAR_SPEED=0.05` or `set GB_BACKEND=ollama`.

## Files

```
app.py        web chat UI + API (FastAPI, single page, no internet needed)
gemma.py      system prompt, LiteRT-LM / Ollama / rules backends, JSON parsing
commands.py   action validation, safety limits, background executor with STOP
arm.py        Kinova Gen3 via Kortex API + MockArm
config.py     every setting
scripts/      setup helpers and checks
models/       gemma.litertlm (downloaded)
logs/app.log  every message, Gemma's raw output, latency
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ping 192.168.1.10` fails | Check the cable and the adapter name in the IP script; turn off Wi-Fi if it uses the same subnet |
| `kortex_api` import error / "Descriptors cannot be created directly" | `pip install protobuf==3.20.3` |
| "Arm aborted the move" | The target is unreachable or the arm is in a fault. Clear faults in the Web App, then move closer to Home. |
| A move is "outside the workspace" | This is intended. Widen `WORKSPACE` in `config.py` once you've measured your table. |
| LiteRT-LM GPU error | The app falls back to CPU automatically. Run `check_gemma.py --probe` and set `GB_LITERT_USE_GPU=0`. |
| LiteRT-LM won't load the model | Try `--backend ollama` (`ollama pull gemma4:e4b`; check the tag with `ollama list`) |
| Gemma replies but doesn't move | Check that "Execute arm actions" is ticked, and read `logs/app.log` for Gemma's raw JSON |
| Gripper doesn't move | The Gen3 with a Robotiq 2F-85 uses finger values 0–1. Test with `check_robot.py --move`. |
