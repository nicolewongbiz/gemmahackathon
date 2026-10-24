# EdgeHand: Air-Gapped Edge-AI Tool-Handover Cell

**Technical specification and workflow, v2 (corrected).** A Kinova Gen3 hands an operator the tool they ask for, such as "hand me the Phillips screwdriver". The cell runs 100% offline. Gemma 4 on LiteRT-LM parses the request and names the tools it sees through an Intel RealSense on the arm's wrist. ROS 2 + MoveIt 2 plans the motion, and a Raspberry Pi provides the request, release and stop buttons plus a link watchdog.

This README is the single source of documentation for EdgeHand: spec, setup, run and troubleshooting.

> **Oct 24: the Raspberry Pi is dropped.** Stopping is the Kinova's red E-stop, plus Ctrl-C in `run_kortex`. The Pi sections below (buttons, watchdog, `pi_bridge`, `pi/edgehand_pi.py`) describe optional code that isn't part of the demo, and `require_pi` now defaults to false. The plan is a laptop (vision fallback, motion, safety) plus a Jetson (Gemma voice, then vision) on the event link (§3.2).

## What changed from v1

| # | v1 said | Problem | v2 |
|---|---|---|---|
| 1 | Pi = "hardware E-stop" | A button read by software over USB is not an E-stop: it fails if the link, DDS or the workstation hangs | The Kinova's own red E-stop is the only E-stop. The Pi gives a **STOP request** button plus a heartbeat watchdog that halts the arm if the Pi link drops |
| 2 | Grasp the **handle**, filter out the shaft | The operator would be handed the metal shaft | The robot grasps the **shaft** near the ferrule and presents the **handle** to the operator |
| 3 | "Low-gain torque compliant state" lets the operator pull the tool free | The gripper is still clamped, so joint compliance releases nothing. ros2_kortex also has no torque or admittance mode through MoveIt | The arm holds position and **detects the tug** from the joint torques it already reports (`/joint_states` effort, N·m), then opens the gripper. Pi button B also releases |
| 4 | Jetson computes a "3D handle pose" from a USB camera | A single camera without depth gives pixels, not metres in the arm's frame. There was no calibration step | The wrist camera looks down from a fixed **look pose**. 4 AprilTags at measured base-frame positions give a pixel→table homography, recomputed in every frame, so camera mounting and arm pose errors don't matter. Tools lie flat, so the pose is (x, y, table height, yaw) |
| 5 | YOLO detects "Phillips screwdriver" | No stock YOLO class for screwdrivers, let alone Phillips vs flathead | Geometry finds every tool (a colourful grip with a metal end). **Gemma 4 names them** from numbered marks plus a description of each tool in words. Handle colour is the fallback. A fine-tuned YOLO pose model is the next step |
| 6 | Launch `kortex_bringup` **and** MoveIt | The MoveIt launch already starts `ros2_control_node` with the Kinova hardware interface. Two launches = two drivers on one arm | Launch only `kinova_gen3_7dof_robotiq_2f_85_moveit_config robot.launch.py` |
| 7 | ROS 2 "Humble/Iron" on the workstation | Iron is end-of-life. MoveIt 2 needs Linux, and this team's laptop setup is Windows | **Humble on Ubuntu 22.04** (dual-boot the Legion) |
| 8 | Pi 4 over USB-C gadget | Pi 4's USB-C port is also its power input. A laptop port rarely gives the 3 A it needs, so it browns out. ROS 2 on Raspberry Pi OS also has no Humble binaries | Pi on its own 5 V / 3 A supply, wired to a USB-Ethernet adapter on the workstation. The Pi runs plain Python and sends UDP; `pi_bridge` on the workstation turns that into ROS topics |
| 9 | Pose targets for "the tool" | MoveIt's `manipulator` group ends at `end_effector_link` (the flange), ~17 cm behind the fingertips, and the 2F-85 fingertips move ~13 mm further out as they close | Targets are computed for the flange from the grasp point and `TCP_OFFSET_M` = closed fingertip distance + `TABLE_CLEARANCE_M` − tool radius, and refused if the closed fingertips would come closer than that to the table (§2.3.1). With 0.15 the sim drove the pads into the table |
| 10 | Camera on the Jetson | The camera on this cell is on the arm's wrist | The arm moves to the look pose before every frame and waits there between handovers |
| 11 | MoveIt plans "collision-free" | MoveIt only avoids what's in its planning scene, and by default that has no table and no tools | `handover_node` adds the table and a keep-out box over the tool tray at startup. Only the short vertical descend/lift run with collision checking off |
| 12 | Vision publishes a pose; the planner uses it | Which frame is fresh? With no network time on an air-gapped cell, timestamps from two machines can't be compared | `handover_node` sends `/edgehand/locate` with an id after the arm has settled at the look pose. `vision_node` answers with the same id, from a frame it received after the request |

### Measured on the real arm (Oct 24)
A read-only Kortex probe (no motion) and checks on the cell found:

- **State:** `ARMSTATE_SERVOING_READY`, 7 actuators, Robotiq 2F-85 on the wrist interconnect.
- **Tool frame:** z = **0.120 m** (tool mass 0.831 kg). The look pose and the GemmaHackathon sim results assume this value.
- **No Kinova Vision module:** no `VISION` device is reported and RTSP port 554 is closed. The camera is an **Intel RealSense on the wrist, cabled to the laptop by USB**.
- **Joint torque noise at rest:** about **0.01 N·m** per joint, so `TUG_TORQUE_NM = 1.5` is far above it. It can probably go down to ~0.5 N·m for a gentler tug.

## 1. System overview

```
     wrist: Intel RealSense ── USB 3 (cable along the arm) ──┐
                                                             v
                         +--------------------------------------------+
                         |   WORKSTATION  Lenovo Legion 5i            |
                         |   Ubuntu 22.04, ROS 2 Humble, MoveIt 2     |
                         |   intent_node   Gemma 4 E4B: request → tool|
                         |   vision_node   wrist camera, tags, Gemma  |
                         |                 names the tools            |
                         |   handover_node state machine + MoveIt     |
                         |   pi_bridge     UDP <-> ROS, watchdog      |
                         +--------------------------------------------+
                  RJ45  /                                          \  USB-Ethernet adapter
     192.168.1.11      /                                            \  192.168.60.1
                      v                                              v
         +------------------+                               +--------------------+
         | KINOVA GEN3      |                               | RASPBERRY PI 4     |
         | + Robotiq 2F-85  |                               | own 5 V / 3 A PSU  |
         | 192.168.1.10     |                               | REQUEST, RELEASE,  |
         | red E-stop =     |                               | STOP buttons       |
         |  THE e-stop      |                               | UDP, no ROS        |
         +------------------+                               | 192.168.60.2       |
                                                            +--------------------+
```

| Device | Link to workstation | Addresses | Job |
|---|---|---|---|
| Kinova Gen3 + 2F-85 | Built-in RJ45 | arm 192.168.1.10, laptop 192.168.1.11 | Executes trajectories (ros2_kortex, low-level servoing), reports joint torques |
| Intel RealSense (wrist) | USB 3, cable along the arm | OpenCV index `CAMERA_USB_INDEX` | Colour frames from the look pose (it shows up as a UVC webcam; no RealSense driver needed) |
| Raspberry Pi 4 | USB-Ethernet adapter on the laptop | laptop 192.168.60.1, Pi 192.168.60.2 | 3 buttons + 10 Hz heartbeat over UDP :5005 |
| Jetson (optional) | USB-C device mode | Jetson 192.168.55.1 | To give it the vision job, plug the RealSense into the Jetson and run `vision_node` there. The locate request is id-based, so the machines' clocks don't matter |

There's no switch and no Wi-Fi.

## 2. Software

### 2.1 Intent: Gemma 4 on LiteRT-LM (workstation)
- `intent_node` subscribes to `/edgehand/command` (`std_msgs/String`, the operator's text) and runs Gemma 4 E4B (`gemma-4-E4B-it.litertlm`) through the LiteRT-LM Python API. Text only, about 1.5 s per call.
- Gemma returns one JSON object: `{"tool": "phillips" | "flathead" | "torx" | null, "reply": "..."}`. The code extracts the first `{...}`, whitelists `tool` against `TOOLS`, and retries twice. An unknown tool becomes `null` and the arm doesn't move. If the operator describes a job instead of naming a tool, the prompt tells Gemma to pick the tool that does it.
- **Gemma never outputs coordinates.** It names a tool class. Geometry comes from calibrated vision.

### 2.2 Perception: wrist camera (workstation)
- **Camera:** the RealSense on the wrist, read with OpenCV as a USB webcam (`CAMERA_SOURCE = "usb"`). `"rtsp"` is for an arm with a Kinova Vision module and `"topic"` for a ROS image topic.
- **Look pose:** frames are only used with the arm at `LOOK_FLANGE_XYZ` / `LOOK_THETA`, gripper roughly straight down and high over the tray. The default is the pose measured on the real cell (Oct 24): Web App tool pose (0.266, −0.028, 0.447), θ (174.3, −7.2, 89.8), with all 4 tags in view and CALIBRATED on the real RealSense. The slight tilt is absorbed by the tag homography. Re-check the view if the tray or tags move (Phase 1.5).
- **Calibration:** 4 AprilTags (tag36h11, IDs 0–3, 5 cm) at the tray corners, centres measured in the robot base frame (`TABLE_TAGS`). Every frame: tags → homography pixel → table (x, y). Tools lie flat, so z = table height + tool radius.
- **Finding tools** (`vision.find_tools`): any colourful grip with a grey, unsaturated part sticking out of one end, on a dark mat. Grips are split by hue, so touching tools of different colours stay separate. The metal end is the shaft. Black grips are invisible on a dark mat, and stacked tools aren't supported.
- **Naming tools** (`namer.GemmaNamer`): Gemma 4 E4B sees the tray image with each tool's number drawn on its handle, plus `TOOL_DESCRIPTIONS` ("screwdriver with a red handle"), and returns `{"1": "phillips", "2": "unknown"}`. Names are whitelisted. `VISION_NAMER = "colour"` names by handle colour (`TOOLS` HSV ranges) instead.
- **Grasp target:** on the **shaft**, `GRASP_FROM_FERRULE_M` (2 cm) from where the handle ends. The yaw is along the tool axis, pointing **towards the handle**.
- **Answer:** `/edgehand/locate_result` `{"id", "tool", "found", "grasp_xy", "handle_dir", "seen"}`. If the tool isn't there, `handover_node` says what it can see. `/tool_pose_target` (PoseStamped) is also published for RViz.

### 2.3 Motion (workstation): ROS 2 Humble + MoveIt 2
- One launch brings up the driver, controllers, `move_group` and RViz: `kinova_gen3_7dof_robotiq_2f_85_moveit_config robot.launch.py`.
- `handover_node` plans with the `move_group` action (`manipulator` group, tip `end_effector_link`) for free-space moves, and with `/compute_cartesian_path` for straight approach and lift, executing through `/execute_trajectory`. All plans use velocity and acceleration scaling 0.2. Humble's Cartesian service has no scaling field, so those trajectories are slowed in time.
- Flange targets = grasp point − `TCP_OFFSET_M` along the tool z axis. The gripper points straight down for picking, with the fingers (`FINGERS_ALONG`) closing across the shaft.
- At startup `handover_node` adds the table and a `TRAY_KEEPOUT_H` box over the tag rectangle to the planning scene (`/apply_planning_scene`). Free moves stay out of the tray; only the vertical descend/lift use `avoid_collisions: false`.
- Gripper: `/robotiq_gripper_controller/gripper_cmd` (`control_msgs/GripperCommand`, 0.0 open … 0.8 closed).
- **Missed grasps:** after the lift, a finger position (`GRIPPER_JOINT` in `/joint_states`) past `GRIPPER_EMPTY_POS` means nothing is between the fingers. The arm opens, goes back to the look pose, asks vision again and retries, up to `PICK_ATTEMPTS` picks, then says it missed. Off with `use_fake_hardware:=true` (mock fingers always close fully).

### 2.3.1 Never hitting the table
The lowest the gripper ever gets is its closed fingertips (the 2F-85 tips move ~13 mm further out as they close). The layers:

1. **Clearance.** Every grasp puts the closed fingertips `TABLE_CLEARANCE_M` (6 mm) above the table. `plan_targets` computes the fingertip height of every target and refuses the whole pick, before any motion, if one would be lower.
2. **Table height measured with the fingertips.** `TABLE_TOUCH_Z` (Phase 1.6) holds the flange height with the closed fingertips touching each tag. The plane through them follows a tilted table, and because the same arm measured it, ruler error in `FINGERTIP_CLOSED_M` and the arm's own z offset cancel out of the grasp height. A 1° tilt across the tray is 4 mm, almost the whole margin, so don't skip this.
3. **Planning-scene table 6 mm above the real one,** with a hole around the base, so `move_group` keeps every link that high on free moves. (Before, its top was 5 mm *below* the table so the base wouldn't collide.)
4. **Slow last 3 cm.** The descent switches to `DESCEND_SLOW_SCALE` (5%) for the final `DESCEND_SLOW_M`, so a wrong height touches gently with time to hit the E-stop.
5. **Air run first** (`AIR_GAP_M = 0.03`, Phase 3): the whole sequence runs with every grasp 3 cm above the tools.

Measured in MuJoCo (§6): the lowest gripper point is 0.7 mm below the nominal clearance (pad shape), so 5.3 mm with the default. Picks work the same from 4 to 10 mm clearance in sim; above ~6 mm the pads hold the shaft with only their bottom edge, which a real gripper is less forgiving of.

### 2.4 Handover and release
- `HANDOVER_XYZ` is a fixed point at the operator's chest height, away from their face. The tool is rotated so the handle points along `OPERATOR_DIR` (towards the operator).
- Once the arm is still, `handover_node` records the joint torques as a baseline. A tug changes the wrist torques. When the change stays above `TUG_TORQUE_NM` for `TUG_SAMPLES` samples, the gripper opens. Pi button **RELEASE** (B) opens it too.
- With `use_fake_hardware:=true` the efforts are zero, so only the button releases.

### 2.5 Safety
- The **Kinova red E-stop** is the only emergency stop. Someone keeps a hand on it.
- Pi **STOP** → `/edgehand/stop` → `handover_node` cancels the running trajectory (the controller holds position). A stop during the look move or while vision is looking also ends the request. New requests are refused for 1 s after a stop.
- **Watchdog:** if no Pi heartbeat arrives for `WATCHDOG_S` (0.5 s), `pi_bridge` publishes `/edgehand/stop` at 5 Hz until it's back.
- Speed: velocity and acceleration scaling 0.2. Every target is checked against `WORKSPACE` before planning.
- Route the RealSense cable with slack at every joint; joint 7 (the wrist) rotates.

### 2.6 Topics

| Topic | Type | From → to |
|---|---|---|
| `/edgehand/command` | `std_msgs/String` | operator text (CLI / UI) → `intent_node` |
| `/edgehand/tool_request` | `std_msgs/String` | `intent_node`, `pi_bridge` (REQUEST button = `DEFAULT_TOOL`) → `handover_node` |
| `/edgehand/locate` | `std_msgs/String` (JSON `{"id", "tool"}`) | `handover_node` (arm at the look pose) → `vision_node` |
| `/edgehand/locate_result` | `std_msgs/String` (JSON) | `vision_node` → `handover_node` |
| `/tool_pose_target` | `geometry_msgs/PoseStamped` | `vision_node` → RViz |
| `/edgehand/detections` | `std_msgs/String` (JSON) | `vision_node` → debugging |
| `/edgehand/release` | `std_msgs/Empty` | `pi_bridge` (RELEASE button) → `handover_node` |
| `/edgehand/stop` | `std_msgs/Empty` | `pi_bridge` (STOP button, watchdog) → `handover_node` |
| `/edgehand/state`, `/edgehand/say` | `std_msgs/String` | `handover_node`, `intent_node` → UI / logs |

## 3. Workflow

```
 REQUEST button (Pi)  or  "hand me the Phillips" (text)
            |                           |
            |                  intent_node: Gemma 4 → {"tool": "phillips"}
            v                           v
      /edgehand/tool_request = "phillips"
            |
            v
 handover_node: MoveIt → look pose (gripper down over the tray), settle 0.5 s
            |   /edgehand/locate {"id": 7, "tool": "phillips"}
            v
 vision_node: next wrist-camera frame → tags → homography → find tools → Gemma names #1, #2, ...
            |   /edgehand/locate_result {"id": 7, "found": true, "grasp_xy", "handle_dir"}
            |   (not found → "I can't see the phillips. I can see flathead and torx.")
            v
 handover_node, MoveIt at 20 % speed:
   open → above tool → straight down → close → straight up
        → HANDOVER_XYZ, handle towards the operator → hold still, record torque baseline
            |
            v
   tug detected (Δ wrist torque)  or  RELEASE button  →  open gripper → retreat → look pose
            |
   STOP button or Pi link lost at any point → cancel trajectory, hold position
```

## 4. Roadmap

### Phase 0: Prerequisites (needs internet)
- **Workstation:** Ubuntu 22.04 (dual boot), then ROS 2 Humble (`ros-humble-desktop`) and the Kinova packages:
  ```bash
  sudo apt install ros-humble-kortex-bringup ros-humble-kinova-gen3-7dof-robotiq-2f-85-moveit-config \
                   ros-humble-moveit ros-humble-control-msgs
  pip install litert-lm-api opencv-python numpy        # system python3.10; litert-lm ships py3 wheels
  ```
  If an apt package is missing, build ros2_kortex from source (`humble` branch, see its README).
  Copy the Gemma model to `EdgeHand/models/gemma-4-E4B-it.litertlm` (it's in `GemmaHackathon/data/`, or `hf download litert-community/gemma-4-E4B-it-litert-lm gemma-4-E4B-it.litertlm --local-dir EdgeHand/models`).
- **Pi 4:** Raspberry Pi OS, then `pip install gpiozero`. Buttons: REQUEST GPIO17, RELEASE GPIO27, STOP GPIO22 (each to GND).
- Build the workspace on the workstation:
  ```bash
  cd EdgeHand/ros2_ws && colcon build --symlink-install && source install/setup.bash
  ```
- Print `GemmaHackathon/tags/apriltags.pdf` at 100% (tags 0–3 are the tray corners).

### Phase 1: Network, camera and calibration
1. **Arm link.** The workstation gets 192.168.1.11/24 on the port wired to the arm's base. macOS: `sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.1.11 255.255.255.0`. Ubuntu: a static IPv4 in Settings or `nmcli`. Check: `ping 192.168.1.10`, then open http://192.168.1.10 (admin / admin).
2. **Pi link.** 192.168.60.1/24 on the second USB-Ethernet adapter and 192.168.60.2/24 on the Pi. Check with `ping 192.168.60.2`.
3. **Camera.** Plug the RealSense into the workstation (USB 3 cable, along the arm with slack at every joint). Check with `python -m edgehand.camera --list`, which prints each index with its name, saves a snapshot per index to `/tmp`, and shows which one will be used. On macOS the camera is picked by name (`CAMERA_USB_NAME`, needs `pip install pyobjc-framework-AVFoundation`), because the numbers shift when an iPhone connects. OpenCV numbers cameras sorted by unique ID, which put the RealSense at 0 and the MacBook camera at 1 on the cell. Elsewhere set `CAMERA_USB_INDEX`. If `system_profiler SPUSBHostDataType` shows the RealSense at 480 Mb/s, it's on USB 2: use a USB 3 cable and plug it in firmly.
4. **Tags and table height, in one pass.** Tape tags 0–3 at the tray corners, flat on the dark mat. Run `python -m edgehand.measure_tags`. It's read-only: you jog the arm, it never moves it. For each tag, close the gripper, point it straight down, jog until the closed fingertips just pinch a sheet of paper on the tag's centre, and press ENTER. It saves `TABLE_TAGS` (fingertip x, y) and `TABLE_TOUCH_Z` (flange z) to `data/calibration.json`, asking first and keeping the old file with a timestamp. It prints the tag rectangle's size to check against a tape measure, and warns if the gripper was more than 3° off vertical.
5. **Look pose.** Jog the arm until `python -m edgehand.camera` shows **CALIBRATED** with all 4 tags outlined and clear of the fingers, then run `python -m edgehand.measure_tags --look`, which saves `LOOK_FLANGE_XYZ` / `LOOK_THETA` to the same file. A small tilt is fine, because the tags give the mapping in every frame. On the cell (Oct 24) that was flange (0.254, −0.043, 0.566), θ (174.3, −7.2, 89.8). Straight down above about z 0.50 (tool frame) is out of reach.
6. **Gripper geometry** (the sim found both of these matter):
   - Close the gripper and measure the flange face → fingertip distance *d* (ruler, or `ros2 run tf2_ros tf2_echo end_effector_link robotiq_85_left_finger_tip_link` with the gripper closed). Set `FINGERTIP_CLOSED_M = d`; `TCP_OFFSET_M` follows from it.
   - **Table height, with the fingertips** (§2.3.1). Gripper closed, pointing straight down (θ 180, 0, 90), a sheet of paper on tag 0. Jog down in small steps until the paper is pinched under the fingertips, then read the flange z: `ros2 run tf2_ros tf2_echo base_link end_effector_link`, or the Web App's z + 0.12 (it shows the tool frame). Repeat on tags 1–3 and enter all four in `TABLE_TOUCH_Z`. If one differs from the others by more than a few mm, the mat isn't flat: fix that first.
   - **Empty vs holding.** Close the gripper on nothing, then on the thinnest shaft you'll use, and read `GRIPPER_JOINT` each time (`ros2 topic echo /joint_states`). Check the joint name is there, and set `GRIPPER_EMPTY_POS` halfway between the two (sim: 0.782 empty, 0.745 on a 6 mm shaft).
   - Open and close the gripper in RViz with the `end_effector_link` axes shown: set `FINGERS_ALONG` to the flange axis ("x" or "y") the fingers move along.
7. **Tool descriptions.** Write what each real tool looks like from above in `TOOL_DESCRIPTIONS`, e.g. "screwdriver with a red and black handle".
8. **Shoot test frames** (no arm motion; the arm stays at the look pose). Run `python -m edgehand.label --capture --note "<scene>"`, press SPACE to save a frame, then label it: press the tool's class number, click its grasp point on the shaft, then click towards the handle. ENTER saves. Shoot each of these:
   - room light
   - side lamp
   - flashlight glare on the mat
   - two tools touching
   - a non-tool in the tray
   - the empty tray (press ENTER without labelling)

   Frames go to `data/frames/` at 1280×720, JPEG quality 70, like the stream. `python -m pytest -s test/test_real_frames.py` then checks real tools and lighting without the arm: every labelled tool found, grasp within 10 mm, handle direction right, colour class and stray detections reported. Add `EDGEHAND_TEST_GEMMA=1` to check Gemma's names too. The labels are pixels, so re-measuring `TABLE_TAGS` doesn't invalidate them, but the frames only calibrate once `TABLE_TAGS` matches where the tags really are.

**Moving the cell (e.g. to the venue).** Everything is measured relative to the arm's base, so the table's height off the floor doesn't matter. If the arm and the tagged tray surface move together (arm bolted to the breadboard, tags taped on it), the calibration still holds: check that `python -m edgehand.camera` shows CALIBRATED and do one slow pick. If they were separated, re-run steps 4–5 and `measure_tags --gripper`, about 5 minutes, with `--site venue` to label the file. Re-shoot the test frames under the venue's lighting either way (step 8). `config.py` loads `data/calibration.json` (or `$EDGEHAND_CALIBRATION`) automatically. Tests and the MuJoCo sims ignore it, because their scenes have their own geometry.

### Phase 2: Core loop
- **Workstation:** `ros2 launch edgehand workstation.launch.py robot_ip:=192.168.1.10`. This starts the MoveIt launch plus `intent_node`, `vision_node`, `handover_node` and `pi_bridge`. Arguments: `camera:=usb|rtsp|topic`, `namer:=gemma|colour`, `show:=true` (vision window).
- **Pi:** `python3 EdgeHand/pi/edgehand_pi.py --host 192.168.60.1`.
- **Text request:** `ros2 topic pub --once /edgehand/command std_msgs/String "{data: 'hand me the phillips screwdriver'}"`.

### Phase 3: Dry run
1. Mock hardware: `ros2 launch edgehand workstation.launch.py robot_ip:=0.0.0.0 use_fake_hardware:=true`. Use the RELEASE button, because there are no torques.
2. MuJoCo, no ROS needed: from `EdgeHand/`, `../GemmaHackathon/sim/.venv/bin/python -m sim.handover --trials 20` (add `--miss` to aim the first grasp past the tool and check the retry). It runs the wrist camera at the look pose, vision, grasp, handover and tug release on a simulated Gen3, and prints the closest the gripper got to the table (§6).
3. **Air run on the real arm**, E-stop in hand. Every pick closes 3 cm above its tool. Check the fingers end up centred over the shaft and across it, and the gripper height looks right, for tools in every corner of the tray.
   - **Without ROS (fastest, from the Mac the camera is on):** `python -m edgehand.run_kortex --air --tool torx` (§3.1). Then without `--air`.
   - With ROS: `AIR_GAP_M = 0.03` in config, the workstation launch, then `AIR_GAP_M = 0.0`.
4. Real picks, E-stop in hand. Tune `TUG_TORQUE_NM`: `run_kortex` prints the live Δτ while holding; with ROS, `ros2 topic echo /edgehand/state`.

### 3.1 Running without ROS: `run_kortex`
`edgehand/kortex_robot.py` implements the same robot interface `handover.py` uses, on the Kortex API's Cartesian controller (`reach_pose`), and `edgehand/run_kortex.py` is the loop: look pose → camera → vision → Gemma names the tools → handover. Everything in §2.3.1 still applies: the clearance check, the table plane from fingertip touches, the slow last 3 cm, the air gap, the miss-and-retry, and the tug release (Kortex reports actuator torques). What's missing is MoveIt's collision planning, so free moves **rise first, translate at the higher of the two heights, then descend**: lateral motion never happens low over the tray. `run_kortex` refuses to pick on config.py's placeholder calibration; run `measure_tags` first.

```bash
cd EdgeHand/ros2_ws/src/edgehand && source ../../../../GemmaHackathon/.venv/bin/activate    # kortex_api, opencv, litert
python -m edgehand.run_kortex --look-only --show        # look pose + what vision sees, no pick
python -m edgehand.run_kortex --air --tool torx         # the whole sequence 3 cm above the tool
python -m edgehand.run_kortex --tool torx               # one real pick
python -m edgehand.run_kortex                           # type requests; Gemma parses them
```
Ctrl-C sends a Kortex Stop. The Pi buttons aren't wired into this runner.

### 3.2 Surgical phase → instrument, and plugging in other nodes
**Demo Channel A.** `edgehand/phase_player.py` plays a public aneurysm-clipping clip (`~/Desktop/clippingdemo.mp4`, 67 s, owner-enabled download, licence allows reuse) at its own frame rate with the current phase, the instrument it calls for, the last request sent and a phase timeline over it. Each time the phase **changes** to one in `config.PHASE_TOOLS` it sends one `request` to the arm's laptop over the event link (below). **There is no phase model in the demo.** The timeline is a hand annotation, `data/phases/clippingdemo.csv` (`time_s,phase` rows; repeated rows of one phase are a single segment), and the overlay says so: "Phase (annotated)" and "HAND-ANNOTATED phase timeline (not model output)". Never present it as model output.

| Clip time | Phase (annotated) | Request sent at the change | Stand-in tool (`PHASE_TOOLS`, editable) |
|---|---|---|---|
| 0–6 s | Parent vessel exposure | 0 s | phillips |
| 6–18 s | Clipping | 6 s | torx |
| 18–54 s | Dome exposure | 18 s | flathead |
| 54–58 s | Clipping | 54 s | torx |
| 58 s–end | Dome exposure | 58 s | flathead |

Running the demo, from `EdgeHand/ros2_ws/src/edgehand` with the venv of §3.1, two terminals on the laptop the arm is on:
```bash
python -m edgehand.run_kortex --listen --namer first                                        # 1: the arm, taking requests over the link
python -m edgehand.phase_player ~/Desktop/clippingdemo.mp4 ../../../data/phases/clippingdemo.csv   # 2: play + send (SPACE pause, ESC quit)
python -m edgehand.phase_player ~/Desktop/clippingdemo.mp4 ../../../data/phases/clippingdemo.csv --headless   # just list the requests
python -m edgehand.phase_player clip.mp4 phases.csv --host 192.168.55.100 --map "Clipping=flathead"           # from the Jetson, mapping override
```
A handover takes 20–40 s, so a request that arrives while one is running (the 6 s change, six seconds into the handover the 0 s change started; the 18 s one too if that handover is still going) is **dropped by design** with a log line in the runner, never queued; the next change once the arm is free is taken. Checked end to end without the arm: `test/test_phase.py` replays the clip at 8x with `--no-window` against a `link.Listener` and found the five requests at 0.0 / 5.9 / 17.9 / 53.8 / 58.3 s in clip time (tolerance 0.5 s); it skips when the clip isn't on the Desktop.

**Making phases.csv from the model (not used in the demo).** `phase/predict_phases.py` runs the neurosurgical phase model (code in [`NeurosurgicalPhaseRecognition/`](../NeurosurgicalPhaseRecognition/), from [veronicawang1/neurosurgical_phase_recognition](https://github.com/veronicawang1/neurosurgical_phase_recognition), aneurysm clipping, 4 phases: Brain Exposure, Parent Vessel Identification, Dome and Neck Identification, Clipping) on the clip. It follows the repo's pipeline: 1 FPS frames, fine-tuned ResNet50 features, MS-TCN (2×8 layers, 128 filters), then Viterbi with the repo's surgical-ordering prior, because the learned transition matrix needs the training labels. It needs the two trained weight files from the training VM:
```bash
uv venv phase/.venv --python 3.11 && uv pip install --python phase/.venv/bin/python -r phase/requirements.txt
phase/.venv/bin/python phase/predict_phases.py --repo ../NeurosurgicalPhaseRecognition --video ~/Desktop/clippingdemo.mp4 \
    --backbone <finetuned_backbone.pt> --mstcn <best_mstcn.pt> --out data/phases/clippingdemo_model.csv   # ~40 s on an M-series Mac
# play that one with --model-output, so the overlay labels it as model predictions instead of annotation
```

**Other nodes** (`edgehand/link.py`). Small JSON events over UDP port `EVENT_UDP_PORT` (5006) to the laptop, which is the only computer that commands the arm: `request` (tool, source, phase), `release`, `stop`, and a `heartbeat` twice a second. Nodes send events, never motions. Plugging in the Jetson's USB-C gives the laptop 192.168.55.100 with no setup. A node that isn't plugged in just isn't heard, and the laptop carries on with typed requests. `run_kortex --listen` turns the listener on.

### 3.3 Voice on the laptop, and setting up the Jetson
**Voice** (`edgehand/voice_node.py`, on the laptop with its mic). Press ENTER, speak for `VOICE_SECONDS` (4 s), and Gemma 4 E4B, using its native audio input, returns `{"intent": request|stop|release|none, "tool", "heard", "reply"}`. The tool is checked against `TOOLS`, and the result goes to `run_kortex --listen` over the event link. The reply is spoken with macOS `say`, offline. The mic is picked by name (`VOICE_MIC_NAME`), because AVFoundation indexes shift when an iPhone connects. A spoken "stop" is a convenience; the red E-stop is the stop. Checked with synthesized speech: 4/4 phrases right, 1.9–3.6 s each.
```bash
python -m edgehand.voice_node            # ENTER, speak       --text: type instead       --list-mics
```
**Jetson** (Orin Nano Super, JetPack 6.2 / L4T R36.4.7, Ubuntu 22.04, user `nicole`, USB-C → 192.168.55.1). A separate venv keeps the system Python untouched. The system OpenCV is 4.5.4, which has no `ArucoDetector`.
```bash
python3 -m venv ~/edgehand-venv && ~/edgehand-venv/bin/pip install litert-lm-api opencv-python-headless numpy
mkdir -p ~/edgehand-models   # copy gemma-4-E4B-it.litertlm here (3.7 GB, ~2 min over USB-C)
# code: copy EdgeHand/ros2_ws/src/edgehand/{edgehand,test} to ~/edgehand-src (kept apart from ~/edgehand/cam_relay.py)
```
LiteRT-LM's aarch64 `liblitert-lm.so` loads, with CPU, GPU and NPU backends listed. Session 38's `cam_relay.py` serves the RealSense colour stream from the Jetson on :8089 (`CAMERA_SOURCE = "url"`), because macOS gives only one of the D435's two video functions (infrared or colour) per plug-in.

### 3.4 The surgical assistant demo (Jetson video → Gemma on the laptop → arm)
```
Jetson: surgical_feed.py                    Laptop: assistant.py (Gemma 4 E4B, LiteRT-LM, GPU)       run_kortex --listen
 clip plays as the "microscope camera"  ──►  frame + phase + tray, one Gemma call:                ──► fetches the instrument
 MJPEG :8090, phase events -> :5007          "sees 3 instruments: forceps, clip applier, ..."
 (hand-annotated timeline)                   -> green clip applier: "clipping phase needs ..."
```
- **Stand-ins** (`INSTRUMENTS`): green = clip applier, orange cap = suction, yellow collar = bipolar forceps, clear blue = micro-dissector.
- **What's live and what isn't:** the phase is the hand-annotated timeline, because the phase model's weights aren't available, and the screen says so. What Gemma sees in the frame, its choice and its reason are live inference.
- **Two Gemma calls per phase change.** (1) **Scene**, with the image, and Gemma is *not* told the phase: instruments in view, aneurysm visible, clip applied, and Gemma's own read of the phase. (2) **Decision**, text only: that scene report, plus the annotated phase, plus the tray, gives the instrument and a reason. Because the read is independent, the on-screen agreement means something. Measured through the Jetson: 6.3–8.0 s for the scene plus 1.6–2.0 s for the decision. Gemma's zero-shot phase read agreed with the annotation 3/7 times (it leans towards "Dome exposure"), against the trained phase model's 95.7%. That's the case for a specialised phase model supplying the phase while Gemma reasons over the scene. A future version could feed Gemma's scene features into the MS-TCN alongside the CNN features.
- **Spoken confirmation** (`ASSISTANT_CONFIRM`, on by default; `--no-confirm` turns it off). The laptop says the recommendation out loud with macOS `say`, which is offline ("Clipping. Shall I bring the green clip applier?"), then listens for 3.5 s, no key press needed. "Yes" fetches it, "no" skips it, "no, the suction" fetches that instead, and stop/release work as always. **The arm never moves on Gemma's pick without a spoken yes.** Yes, no, stop and release come from fixed keywords on a neutral transcription. Gemma only maps a named instrument, because with the phase and tray in the prompt it once heard "stop" as "clip applier". Synthesized check: yes, no, redirect, stop and a non-answer all handled, in 0.9–2.6 s.
- **Voice any time:** press **v** in the assistant window to request an instrument, stop or release.
- **GPU:** during the demo the assistant owns the laptop GPU, and run_kortex names tray tools by colour (`--namer colour`). Only one LiteRT GPU process runs at a time.
- **Measured** on the clip through the Jetson: 6.4–8.3 s per decision on the GPU (16–26 s on the CPU). All five phase changes were decided, with picks blue micro-dissector, then green clip applier, then blue, then green.
```bash
ssh nicole@192.168.55.1 '~/edgehand-demo/start_feed.sh'            # Jetson: deploy/jetson_start_feed.sh (loops; --speed 0.5 to slow)
python -m edgehand.run_kortex --listen --namer colour               # laptop, terminal 1 (the arm)
python -m edgehand.assistant                                         # laptop, terminal 2 (the stage screen)
python -m edgehand.assistant --local ~/Desktop/clippingdemo.mp4 ../../../data/phases/clippingdemo.csv   # no Jetson
```
Phases change faster than a handover (20–40 s), so `run_kortex` never queues requests: while a handover runs it keeps only the **latest** request pending (each newer one replaces it; one for the tool being handed over is dropped) and runs it once the operator has taken the tool. A 67 s clip therefore fits one or two handovers; slow it with `--speed`, or loop it.

## 5. Settings

Every tunable is in `ros2_ws/src/edgehand/edgehand/config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `TOOLS` | phillips / flathead / torx | tool classes; HSV handle colours for `VISION_NAMER = "colour"` |
| `TOOL_DESCRIPTIONS` | "screwdriver with a red handle", … | what Gemma is told each tool looks like from above |
| `VISION_NAMER` | "gemma" | "gemma" or "colour" |
| `DEFAULT_TOOL` | phillips | what the REQUEST button asks for |
| `CAMERA_SOURCE`, `CAMERA_URL` | "url", http://192.168.55.1:8089/stream.mjpg | wrist camera: "url" = `cam_relay.py` on the Jetson serving the RealSense as MJPEG over the USB link (macOS only ever shows one of the D435's IR/colour cameras, chosen by a race at plug-in; Linux lists the colour one every time); "usb" (RealSense on this computer, by name), "rtsp", "topic" |
| `LOOK_FLANGE_XYZ`, `LOOK_THETA` | (0.254, −0.043, 0.565), (174.3, −7.2, 89.8) | the look pose, measured on the real cell Oct 24 (Phase 1.5). The sim uses its own (0.40, 0, 0.62) because its camera mount differs |
| `LOOK_SETTLE_S`, `LOCATE_SKIP_FRAMES` | 0.5, 5 | wait at the look pose; frames dropped after a locate request |
| `TABLE_TAGS` | measured | tag centres (m, base frame) |
| `TABLE_TOUCH_Z` | {} (measure) | flange z with the closed fingertips touching each tag (Phase 1.6); a plane through them is the table. Empty = flat at `TABLE_Z` |
| `TABLE_CLEARANCE_M` | 0.004 | closed fingertips above the table at the bottom of every pick; lower targets are refused (§2.3.1). 4 mm relies on `TABLE_TOUCH_Z` measured with the fingertips (the cell's 4 touches agreed within 2.5 mm) |
| `FINGERTIP_CLOSED_M` | 0.173 | flange → closed fingertips (Phase 1.6) |
| `AIR_GAP_M` | 0.0 | 0.03 for the first real runs: every grasp 3 cm above the tools, no grasp check |
| `DESCEND_SLOW_M`, `DESCEND_SLOW_SCALE` | 0.03, 0.05 | the last 3 cm of each descent at 5% speed |
| `GRIPPER_JOINT`, `GRIPPER_EMPTY_POS` | robotiq_85_left_knuckle_joint, 0.765 | finger position past this after the lift = missed grasp (Phase 1.6) |
| `PICK_ATTEMPTS` | 3 | picks per request; after a miss the arm looks again |
| `BASE_HOLE_M` | 0.08 | half-width of the hole in the planning-scene table around the base |
| `GRASP_SIDE` | "handle" | where the fingers close: "handle" = 2 cm into the handle from the ferrule (precision screwdrivers: the 3–4 mm shaft lies below the fingertips' clearance; the operator gets the handle's free end), "shaft" = 2 cm down the shaft (fat-shafted tools; the sim and the synthetic tests) |
| `TOOL_RADIUS_M` | 0.005 | z of the grasp above the table: radius of the part gripped |
| `GRASP_FROM_FERRULE_M` | 0.02 | grasp point, measured from where the handle ends, on `GRASP_SIDE` |
| `TCP_OFFSET_M` | 0.167 (derived) | flange → grasp point along tool z = `FINGERTIP_CLOSED_M + TABLE_CLEARANCE_M − TOOL_RADIUS_M` |
| `FINGERS_ALONG` | "x" | flange axis the fingers close along (Phase 1.6) |
| `TRAY_KEEPOUT_H` | 0.06 | height of the planning-scene box over the tool tray |
| `SHAFT_MIN_VAL`, `SHAFT_MAX_SAT` | 140, 70 | HSV: a shaft is bright and unsaturated; a lit dark mat must stay below `SHAFT_MIN_VAL` |
| `SEGMENTATION` | "contrast" | "contrast": anything unlike the mat, handle = the wide end (any colour; needs a plain mat the tools stand out from). "colour": colourful grip + grey shaft on a dark mat |
| `CONTRAST_MIN`, `WARP_MARGIN_M` | 45, 0.08 | how unlike the mat a pixel must be; how far past the tags the top-down view reaches (handles overhang) |
| `TOOL_MIN_AREA_M2`, `MIN_ELONGATION`, `MIN_SHAFT_BEYOND_GRASP_M` | 3 cm², 2.5, 0.015 | what counts as a tool: size, long and thin, room on the shaft for the fingers |
| `GRIP_MIN_SAT`, `GRIP_MIN_VAL` | 90, 60 | HSV: what counts as a colourful grip (`"colour"` only) |
| `HOVER_M` | 0.12 | approach height above the grasp |
| `HANDOVER_XYZ`, `OPERATOR_DIR` | (0.45, 0.30, 0.35), (0, 1, 0) | where the tool is presented and which way the handle points |
| `TUG_TORQUE_NM`, `TUG_SAMPLES` | 1.5, 5 | release threshold on wrist-joint Δτ (measured noise ~0.01 N·m) |
| `VEL_SCALE`, `ACC_SCALE` | 0.2, 0.2 | MoveIt scaling |
| `WORKSPACE` | x 0.15–0.80, y ±0.55, z 0–0.70 | targets outside are refused |
| `WATCHDOG_S` | 0.5 | Pi heartbeat timeout |

## 6. Code layout and testing

```
EdgeHand/
  README.md                    this spec
  config/fastdds_peers.xml     unicast DDS discovery, if vision runs on the Jetson and multicast over USB fails
  pi/edgehand_pi.py            Pi: buttons + heartbeat → UDP (no ROS)
  ros2_ws/src/edgehand/
    edgehand/config.py         every setting
    edgehand/gemma.py          LiteRT-LM engine + JSON replies (the only place litert_lm is used)
    edgehand/intent.py         request → tool class, whitelisted (no ROS)
    edgehand/camera.py         wrist camera reader: USB or RTSP (no ROS); `python -m edgehand.camera` = live check
    edgehand/vision.py         homography, find_tools (grips + metal end), shaft grasp pose (no ROS)
    edgehand/namer.py          Gemma names the found tools from a numbered tray image (no ROS)
    edgehand/geometry.py       quaternions, Kortex angles, look pose, flange-from-fingertip (no ROS)
    edgehand/release.py        tug detector on joint torques (no ROS)
    edgehand/handover.py       state machine over a small robot interface (no ROS)
    edgehand/moveit_client.py  MoveGroup / Cartesian path / execute / gripper / planning scene (rclpy)
    edgehand/nodes/*.py        intent_node, vision_node, handover_node, pi_bridge (rclpy)
    launch/workstation.launch.py
    test/test_core.py          pytest for the no-ROS modules (synthetic camera frames, fake robot)
    test/fake_cell.py          fake move_group / controllers / gripper / camera for testing the real nodes
    test/run_integration.sh    all nodes (real Gemma) against fake_cell
    edgehand/kortex_robot.py   the Robot interface on the Kortex API (no ROS): rise-translate-descend free moves
    edgehand/run_kortex.py     no-ROS runner: camera → vision → Gemma → handover on kortex_robot (§3.1)
    edgehand/cam_relay.py      runs ON the Jetson (stock python/opencv): RealSense colour → MJPEG over HTTP :8089
    test/test_kortex_robot.py  kortex_robot with the SDK replaced: angles, move ordering, refusals, units
  sim/handover.py              MuJoCo end-to-end test: wrist camera at the look pose → vision → handover → operator tug
```

**Tested so far** (Oct 23–24, on a Mac with ROS 2 Humble from RoboStack):

| Test | Command (from `EdgeHand/ros2_ws/src/edgehand/`) | Result |
|---|---|---|
| Unit tests | `python -m pytest test/test_core.py` | 33/33 |
| Gemma 4 E4B intent, 7 phrasings ("cross-head", "the flat one", "star", a hammer, off-topic) | `python -m edgehand.intent <model> < phrases.txt` | 7/7 correct, ~1.4 s each |
| Gemma naming tools on synthetic trays (1–3 screwdrivers, plus a blue tool that isn't on the list) | 20 trays | 38/43 named correctly (88%), ~4.6 s per tray. Misses: the unlisted tool called a listed one (3), number mix-ups (2) |
| All nodes vs fake MoveIt, vision from a camera topic: text → Gemma → look → locate → Gemma names → MoveIt → tug → release; STOP mid-move; Pi REQUEST + RELEASE over UDP; a tool that isn't there; a grasp that closes on nothing (looks again, picks again) | `bash test/run_integration.sh` | 5/5 |
| Pi watchdog | heartbeats stop | `/edgehand/stop` within 0.45 s, none while beating |
| MuJoCo, wrist camera at the look pose, random tool poses: vision → pick → present → 12 N tug | from `EdgeHand/`: `../GemmaHackathon/sim/.venv/bin/python -m sim.handover --trials 20` (needs the sim venv and models from `GemmaHackathon/README.md` §10) | 20/20, vision ≤ 4.2 mm, handle towards the operator every time, release 0.1 s after the pull, no table contact, gripper never lower than 5.3 mm above the table |
| MuJoCo, first grasp aimed 12 cm past the tool (closes on air) | `... -m sim.handover --trials 10 --miss` | 10/10 seen as a miss, looked again, picked on the second try; never lower than 5.3 mm |
| MuJoCo multi-tool trays with the table-safety changes, colour naming, intent stubbed | `-m sim.tools` trial() with a stub intent, 18 trays | 18/18 handed over, no contacts, never lower than 5.3 mm (hammer 6.3 mm, pliers / wrench 7.3 mm+) |
| MuJoCo multi-tool trays, real Gemma 4 E4B for intent and naming: 4 tools per tray from 6 classes (adds sim-only pliers, wrench, hammer) | from `EdgeHand/`: `-m sim.tools` | Wrist camera + Gemma naming: **29/30 end to end (97%)**, tools found and named 120/120, grasp + handover 22/22, 0 collisions. Before, with the overhead camera and colour naming: 56/60 (93%). Phrase battery 55/59 (93%) |
| Real camera frames, labelled (Phase 1.8) | `python -m pytest -s test/test_real_frames.py` | Checked on synthetic labelled frames: correct labels pass, a 3 cm label error fails (31 mm). The first real frame (empty tray, room light) fails calibration until `TABLE_TAGS` is measured, because the placeholder positions don't match the taped tags |
| Real arm, read-only (no motion) | Kortex probe | see "Measured on the real arm" above |

In the multi-tool sim the remaining misses are all in intent, for the sim-only pliers ("pass me the grips", "the side cutters", "I need to bend this wire" → null) plus "pry open this paint can lid" for the flathead. A worked example in the intent prompt would probably fix them. The sim tools are coloured shapes under even lighting, so naming real tools on a real camera will be harder than 120/120.

**Not tested yet:** frames from the real RealSense (not plugged in when this was written), Gemma naming on real tools and lighting, `moveit_client.py` against a real `move_group` (only against `fake_cell.py`), and the ros2_kortex controllers. Running LiteRT on the GPU at the same time as other GPU work (a second Gemma process, MuJoCo rendering) made calls fail or time out on the Mac, so run one Gemma-heavy job at a time.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ping 192.168.1.10` fails | Cable into the arm's base RJ45, link light on, static 192.168.1.11 on that adapter (Phase 1.1), Wi-Fi off if it uses 192.168.1.x |
| `vision_node` / `edgehand.camera`: no frames | Wrong `CAMERA_USB_INDEX` (`--list`), USB 2 or charge-only cable, another app holding the camera. On a Mac, allow camera access for the terminal |
| NOT CALIBRATED at the look pose | All 4 tags in view and not hidden by the fingers (Phase 1.5), lighting, print size 5 cm |
| Tools touching or crossing | They merge into one blob and are dropped; the runner prints "tools touching or crossing? separate them". Keep tools a few cm apart, none crossing, all fully on the paper |
| Tool not found | **The mat.** Dark handles (black, dark green, clear) on the black breadboard are invisible: put white paper under the whole tag rectangle. With `SEGMENTATION = "contrast"` anything unlike the mat is a candidate, so the tool must lie entirely on the paper, not touching another tool or a tag, and not touching the frame's edge (the gripper fingers are in every frame there). Shadows under `CONTRAST_MIN`. With `"colour"`: `GRIP_MIN_SAT`, `SHAFT_MIN_VAL`. Check with `run_kortex --look-only --show` or `show:=true` |
| Tool found but named wrong | Make `TOOL_DESCRIPTIONS` say what's visible from above (handle colours and shape). Or `namer:=colour` with tuned `TOOLS` HSV ranges |
| Gemma calls time out | Another process is using the GPU (a second Gemma, a sim); LiteRT gives up after 2 min. Run one at a time |
| Two drivers / controller_manager errors | Only the MoveIt launch should run; kill any `kortex_bringup` launch |
| Arm grips air / hits the tray | `FINGERTIP_CLOSED_M` and `TABLE_TOUCH_Z` (Phase 1.6); do an air run (Phase 3.3) |
| "… puts the fingertips … mm above the table" | A target was refused for being too low: check `TABLE_TOUCH_Z`, `TOOL_RADIUS_M`, and that `TCP_OFFSET_M` wasn't overridden |
| "missed the tool" every time | `GRIPPER_EMPTY_POS` too low, or `GRIPPER_JOINT` not in `/joint_states` (Phase 1.6) |
| Releases on its own | Raise `TUG_TORQUE_NM`, or wait longer before the baseline is taken (`SETTLE_S`) |
| Never releases on a tug | Lower `TUG_TORQUE_NM`; check `/joint_states` efforts aren't zero (they are with fake hardware) |
| Arm stops by itself | Pi watchdog: is `edgehand_pi.py` running, and is `pi_bridge` getting heartbeats? |
| MoveIt "no plan found" | Target outside reach or `WORKSPACE`; move the tray or `HANDOVER_XYZ` closer |

MIT for code written at the event. Gemma models are subject to the Gemma terms of use.
