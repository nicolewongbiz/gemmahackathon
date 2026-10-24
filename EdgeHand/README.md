# EdgeHand: Air-Gapped Edge-AI Tool-Handover Cell

**Technical specification and workflow, v2 (corrected).** A Kinova Gen3 hands an operator the tool they ask for, such as "hand me the Phillips screwdriver". The cell runs 100% offline: Gemma 4 on LiteRT-LM parses the request, a Jetson finds the tool, ROS 2 + MoveIt 2 plans the motion, and a Raspberry Pi provides the request/stop buttons and a link watchdog.

This README is the single source of documentation for EdgeHand: spec, setup, run and troubleshooting.

## What changed from v1

| # | v1 said | Problem | v2 |
|---|---|---|---|
| 1 | Pi = "hardware E-stop" | A button read by software over USB is not an E-stop: it fails if the link, DDS or the workstation hangs | The Kinova's own red E-stop is the only E-stop. The Pi gives a **STOP request** button plus a heartbeat watchdog that halts the arm if the Pi link drops |
| 2 | Grasp the **handle**, filter out the shaft | The operator would be handed the metal shaft | The robot grasps the **shaft** near the ferrule and presents the **handle** to the operator |
| 3 | "Low-gain torque compliant state" lets the operator pull the tool free | The gripper is still clamped, so joint compliance releases nothing. ros2_kortex also has no torque or admittance mode through MoveIt | The arm holds position and **detects the tug** from the joint torques it already reports (`/joint_states` effort, N·m), then opens the gripper. Pi button B also releases |
| 4 | Jetson computes a "3D handle pose" from a USB camera | A single camera without depth gives pixels, not metres in the arm's frame. There was no calibration step | Fixed overhead camera plus 4 AprilTags at measured base-frame positions gives a pixel→table homography. Tools lie flat, so the pose is (x, y, table height, yaw). Calibration is Phase 1 |
| 5 | YOLO detects "Phillips screwdriver" | No stock YOLO class for screwdrivers, let alone Phillips vs flathead. It would need a custom-trained model | Colour-coded handles (red = Phillips, yellow = flathead, green = Torx), segmented with OpenCV. A custom YOLO model is a stretch goal behind the same topic |
| 6 | Launch `kortex_bringup` **and** MoveIt | The MoveIt launch already starts `ros2_control_node` with the Kinova hardware interface. Two launches = two drivers on one arm | Launch only `kinova_gen3_7dof_robotiq_2f_85_moveit_config robot.launch.py` |
| 7 | ROS 2 "Humble/Iron" on the workstation | Iron is end-of-life. MoveIt 2 needs Linux, and this team's laptop setup is Windows | **Humble on Ubuntu 22.04** (dual-boot the Legion). Jetson on **JetPack 6** (Ubuntu 22.04 → Humble from apt) |
| 8 | Pi 4 over USB-C gadget | Pi 4's USB-C port is also its power input. A laptop port rarely gives the 3 A it needs, so it browns out. ROS 2 on Raspberry Pi OS also has no Humble binaries | Pi on its own 5 V / 3 A supply, wired to a USB-Ethernet adapter on the workstation. The Pi runs plain Python and sends UDP; `pi_bridge` on the workstation turns that into ROS topics |
| 9 | Pose targets for "the tool" | MoveIt's `manipulator` group ends at `end_effector_link` (the flange), ~17 cm behind the fingertips, and the 2F-85 fingertips move ~13 mm further out as they close | Targets are computed for the flange from the grasp point and `TCP_OFFSET_M` = closed fingertip distance + 4 mm − tool radius. With 0.15 the sim drove the pads into the table |
| 10 | (missing) | With a fixed overhead camera the arm hides the tools: in Home it hangs right over the tray (found in sim) | The arm **parks** in Kinova's Retract pose (`PARK_JOINTS`) before vision looks and after every handover |
| 11 | MoveIt plans "collision-free" | MoveIt only avoids what's in its planning scene, and by default that has no table and no tools | `handover_node` adds the table and a keep-out box over the tool tray at startup. Only the short vertical descend/lift run with collision checking off |

## 1. System overview

```
                         +--------------------------------------------+
                         |   WORKSTATION  Lenovo Legion 5i            |
                         |   Ubuntu 22.04, ROS 2 Humble, MoveIt 2     |
                         |   intent_node   Gemma 4 E4B (LiteRT-LM)    |
                         |   handover_node state machine + MoveIt     |
                         |   pi_bridge     UDP <-> ROS, watchdog      |
                         +--------------------------------------------+
                  RJ45  /                 | USB-C                 \  USB-Ethernet adapter
     192.168.1.11      /                  | 192.168.55.100         \  192.168.60.1
                      v                   v                         v
         +------------------+   +----------------------+   +--------------------+
         | KINOVA GEN3      |   | JETSON (JetPack 6)   |   | RASPBERRY PI 4     |
         | + Robotiq 2F-85  |   | vision_node          |   | own 5 V / 3 A PSU  |
         | 192.168.1.10     |   | USB cam (overhead)   |   | buttons: REQUEST,  |
         | red E-stop =     |   | 192.168.55.1         |   |   RELEASE, STOP    |
         |  THE e-stop      |   | ROS 2 over DDS       |   | UDP, no ROS        |
         +------------------+   +----------------------+   | 192.168.60.2       |
                                                           +--------------------+
```

| Device | Link to workstation | Addresses | Job |
|---|---|---|---|
| Kinova Gen3 + 2F-85 | Built-in RJ45 | arm 192.168.1.10, laptop 192.168.1.11 | Executes trajectories (ros2_kortex, low-level servoing), reports joint torques |
| Jetson | USB-C device mode (`l4tbr0`) | Jetson 192.168.55.1, laptop 192.168.55.100 (DHCP from the Jetson) | `vision_node`: overhead USB camera → `/tool_pose_target` |
| Raspberry Pi 4 | USB-Ethernet adapter on the laptop | laptop 192.168.60.1, Pi 192.168.60.2 | 3 buttons + 10 Hz heartbeat over UDP :5005 |
| USB camera | USB 3 to the Jetson | `/dev/video0` | Overhead view of the tool tray and tags |

There's no switch and no Wi-Fi. The Jetson and the Pi never talk to each other, only to the workstation, so three separate subnets are fine.

## 2. Software

### 2.1 Intent: Gemma 4 on LiteRT-LM (workstation)
- `intent_node` subscribes to `/edgehand/command` (`std_msgs/String`, the operator's text) and runs Gemma 4 E4B (`gemma-4-E4B-it.litertlm`) through the LiteRT-LM Python API. Text only, about 1–3 s per call.
- Gemma returns one JSON object: `{"tool": "phillips" | "flathead" | "torx" | null, "reply": "..."}`. The code extracts the first `{...}`, whitelists `tool` against `TOOLS`, and retries twice. An unknown tool becomes `null` and the arm doesn't move.
- **Gemma never outputs coordinates.** It names a tool class. Geometry comes from calibrated vision.
- It publishes `/edgehand/tool_request` (`std_msgs/String`, the tool class) and `/edgehand/say` (the reply).

### 2.2 Perception (Jetson)
- **Camera:** a fixed overhead USB camera looking down at the tool tray (dark mat), with 4 AprilTags (tag36h11, IDs 0–3, 5 cm) at the tray corners. Their centres are measured in the robot base frame (`TABLE_TAGS`).
- **Calibration:** in every frame, tags → homography pixel → table (x, y). Tools lie flat, so z = table height + tool radius. There's no depth camera and no hand-eye calibration.
- **Detection:** the dark mat separates tools from the background. Each blob's handle colour gives its class, PCA of the blob gives the tool axis, and the coloured part is the handle while the grey/metal part is the shaft.
- **Grasp target:** on the **shaft**, `GRASP_FROM_FERRULE_M` (2 cm) from where the handle ends. The yaw is along the tool axis, with the pose's +x pointing **towards the handle**.
- **Output:** `/tool_pose_target` (`geometry_msgs/PoseStamped`, frame `base_link`) for the requested class, plus `/edgehand/detections` (JSON list) for debugging.
- **Stretch:** swap the colour classifier for a custom-trained YOLO model. The topic contract stays the same.

### 2.3 Motion (workstation): ROS 2 Humble + MoveIt 2
- One launch brings up the driver, controllers, `move_group` and RViz: `kinova_gen3_7dof_robotiq_2f_85_moveit_config robot.launch.py`.
- `handover_node` plans with the `move_group` action (`manipulator` group, tip `end_effector_link`) for free-space moves, and with `/compute_cartesian_path` for straight approach and lift, executing through `/execute_trajectory`. All plans use velocity and acceleration scaling 0.2.
- Flange targets = grasp point − `TCP_OFFSET_M` along the tool z axis. The gripper points straight down for picking, with the fingers (`FINGERS_ALONG`) closing across the shaft.
- At startup `handover_node` adds the table and a `TRAY_KEEPOUT_H` box over the tag rectangle to the planning scene (`/apply_planning_scene`). Free moves stay out of the tray; only the vertical descend/lift use `avoid_collisions: false`.
- Before vision looks, and after each handover, the arm parks at `PARK_JOINTS` (Kinova Retract), out of the overhead camera's view.
- Gripper: `/robotiq_gripper_controller/gripper_cmd` (`control_msgs/GripperCommand`, 0.0 open … 0.8 closed).

### 2.4 Handover and release
- `HANDOVER_XYZ` is a fixed point at the operator's chest height, away from their face. The tool is rotated so the handle points along `OPERATOR_DIR` (towards the operator).
- Once the arm is still, `handover_node` records the joint torques as a baseline. A tug changes the wrist torques. When the change stays above `TUG_TORQUE_NM` for `TUG_SAMPLES` samples, the gripper opens. Pi button **RELEASE** (B) opens it too.
- With `use_fake_hardware:=true` the efforts are zero, so only the button releases.

### 2.5 Safety
- The **Kinova red E-stop** is the only emergency stop. Someone keeps a hand on it.
- Pi **STOP** → `/edgehand/stop` → `handover_node` cancels the running trajectory (the controller holds position) and refuses new motion until a new request.
- **Watchdog:** if no Pi heartbeat arrives for `WATCHDOG_S` (0.5 s), `pi_bridge` publishes `/edgehand/stop`.
- Speed: velocity and acceleration scaling 0.2. Every target is checked against `WORKSPACE` before planning.

### 2.6 Topics

| Topic | Type | From → to |
|---|---|---|
| `/edgehand/command` | `std_msgs/String` | operator text (CLI / UI) → `intent_node` |
| `/edgehand/tool_request` | `std_msgs/String` | `intent_node`, `pi_bridge` (REQUEST button = `DEFAULT_TOOL`) → `handover_node`, `vision_node` |
| `/tool_pose_target` | `geometry_msgs/PoseStamped` | `vision_node` → `handover_node` |
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
 vision_node: tags → homography, red handle → shaft grasp pose → /tool_pose_target
            |
            v
 handover_node: park out of the camera's view (Retract), take a pose from a frame AFTER parking
 MoveIt, 20 % speed:
   open → above tool → straight down → close → straight up
        → HANDOVER_XYZ, handle towards the operator → hold still, record torque baseline
            |
            v
   tug detected (Δ wrist torque)  or  RELEASE button  →  open gripper → retreat → park
            |
   STOP button or Pi link lost at any point → cancel trajectory, hold position
```

## 4. Roadmap

### Phase 0: Prerequisites (night before, needs internet)
- **Workstation:** Ubuntu 22.04 (dual boot), then ROS 2 Humble (`ros-humble-desktop`) and the Kinova packages:
  ```bash
  sudo apt install ros-humble-kortex-bringup ros-humble-kinova-gen3-7dof-robotiq-2f-85-moveit-config \
                   ros-humble-moveit ros-humble-control-msgs
  pip install litert-lm-api opencv-python numpy        # system python3.10; litert-lm ships py3 wheels
  ```
  If an apt package is missing, build ros2_kortex from source (`humble` branch, see its README).
  Copy the Gemma model to `EdgeHand/models/gemma-4-E4B-it.litertlm` (it's in `GemmaHackathon/data/`, or `hf download litert-community/gemma-4-E4B-it-litert-lm gemma-4-E4B-it.litertlm --local-dir EdgeHand/models`).
- **Jetson:** JetPack 6, then `sudo apt install ros-humble-ros-base ros-humble-v4l2-camera ros-humble-cv-bridge` and `pip install opencv-python pupil-apriltags`.
- **Pi 4:** Raspberry Pi OS, then `pip install gpiozero`. Buttons: REQUEST GPIO17, RELEASE GPIO27, STOP GPIO22 (each to GND).
- Build the workspace on the workstation and the Jetson:
  ```bash
  cd EdgeHand/ros2_ws && colcon build --symlink-install && source install/setup.bash
  ```
- Print `GemmaHackathon/tags/apriltags.pdf` at 100% (tags 0–3 are the tray corners).

### Phase 1: Network and calibration
1. Static IPs. The workstation gets 192.168.1.11/24 on the RJ45 and 192.168.60.1/24 on the USB-Ethernet adapter; the Pi gets 192.168.60.2/24. The Jetson's USB link hands out 192.168.55.100 itself.
2. Check: `ping 192.168.1.10`, `ssh jetson@192.168.55.1`, `ping 192.168.60.2`.
3. On the workstation and the Jetson:
   ```bash
   export ROS_DOMAIN_ID=42
   export ROS_LOCALHOST_ONLY=0
   ```
   If the Jetson's topics don't show up in `ros2 topic list` on the workstation, use the unicast peer profile: `export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/EdgeHand/config/fastdds_peers.xml` on both.
4. **Calibrate the camera.** Tape tags 0–3 at the tray corners. Jog the gripper fingertip onto each tag centre in the Kinova Web App, read x, y and enter them in `TABLE_TAGS`. Also record the table height for `TABLE_Z`.
5. Check: `ros2 run edgehand vision_node --ros-args -p show:=true` shows CALIBRATED and the detected tools, and a clicked tag centre prints metres within 1 cm of `TABLE_TAGS`.
6. **Check the gripper geometry** (the sim found both of these matter):
   - Close the gripper and measure the flange face → fingertip distance *d* (ruler, or `ros2 run tf2_ros tf2_echo end_effector_link robotiq_85_left_finger_tip_link` with the gripper closed). Set `TCP_OFFSET_M = d + 0.004 − TOOL_RADIUS_M`.
   - Open and close the gripper in RViz with the `end_effector_link` axes shown: set `FINGERS_ALONG` to the flange axis ("x" or "y") the fingers move along.
   - Move the arm to `PARK_JOINTS` and check it's out of the overhead camera's view (`show:=true`).

### Phase 2: Core loop
- **Workstation:** `ros2 launch edgehand workstation.launch.py robot_ip:=192.168.1.10`. This starts the MoveIt launch above plus `intent_node`, `handover_node` and `pi_bridge`.
- **Jetson:** `ros2 launch edgehand jetson.launch.py` (`v4l2_camera` + `vision_node`). Check `ros2 topic echo /tool_pose_target`.
- **Pi:** `python3 EdgeHand/pi/edgehand_pi.py --host 192.168.60.1`.
- **Text request:** `ros2 topic pub --once /edgehand/command std_msgs/String "{data: 'hand me the phillips screwdriver'}"`.

### Phase 3: Dry run
1. Mock hardware: `ros2 launch edgehand workstation.launch.py robot_ip:=0.0.0.0 use_fake_hardware:=true`. Use the RELEASE button, because there are no torques.
2. MuJoCo, no ROS needed: from `EdgeHand/`, `../GemmaHackathon/sim/.venv/bin/python -m sim.handover --trials 20` runs vision, grasp, handover and tug release on a simulated Gen3 (§6).
3. Real arm at 20% scaling, E-stop in hand. Tune `TUG_TORQUE_NM` with `ros2 topic echo /edgehand/state` (it prints the live Δτ).

## 5. Settings

Every tunable is in `ros2_ws/src/edgehand/edgehand/config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `TOOLS` | phillips / flathead / torx | handle colour (HSV) per tool class |
| `DEFAULT_TOOL` | phillips | what the REQUEST button asks for |
| `TABLE_TAGS`, `TABLE_Z` | measured | tag centres (m, base frame), table height |
| `TOOL_RADIUS_M` | 0.012 | z of the grasp above the table (shaft/handle radius) |
| `GRASP_FROM_FERRULE_M` | 0.02 | grasp point on the shaft, measured from the handle |
| `TCP_OFFSET_M` | 0.165 | flange → grasp point along tool z: closed fingertip distance + 4 mm − `TOOL_RADIUS_M` (Phase 1.6) |
| `FINGERS_ALONG` | "x" | flange axis the fingers close along (Phase 1.6) |
| `PARK_JOINTS` | Kinova Retract | where the arm waits, out of the camera's view |
| `TRAY_KEEPOUT_H` | 0.06 | height of the planning-scene box over the tool tray |
| `SHAFT_MIN_VAL`, `SHAFT_MAX_SAT` | 140, 70 | HSV: a shaft is bright and unsaturated; a lit dark mat must stay below `SHAFT_MIN_VAL` |
| `HOVER_M` | 0.12 | approach height above the grasp |
| `HANDOVER_XYZ`, `OPERATOR_DIR` | (0.45, 0.30, 0.35), (0, 1, 0) | where the tool is presented and which way the handle points |
| `TUG_TORQUE_NM`, `TUG_SAMPLES` | 1.5, 5 | release threshold on wrist-joint Δτ |
| `VEL_SCALE`, `ACC_SCALE` | 0.2, 0.2 | MoveIt scaling |
| `WORKSPACE` | x 0.15–0.80, y ±0.55, z 0–0.70 | targets outside are refused |
| `WATCHDOG_S` | 0.5 | Pi heartbeat timeout |

## 6. Code layout and testing

```
EdgeHand/
  README.md                    this spec
  config/fastdds_peers.xml     unicast DDS discovery if multicast over USB fails
  pi/edgehand_pi.py            Pi: buttons + heartbeat → UDP (no ROS)
  ros2_ws/src/edgehand/
    edgehand/config.py         every setting
    edgehand/intent.py         Gemma prompt, JSON parsing, whitelist (no ROS)
    edgehand/vision.py         homography, tool segmentation, shaft grasp pose (no ROS)
    edgehand/geometry.py       quaternions, flange-from-fingertip, handover orientation (no ROS)
    edgehand/release.py        tug detector on joint torques (no ROS)
    edgehand/handover.py       state machine over a small robot interface (no ROS)
    edgehand/moveit_client.py  MoveGroup / Cartesian path / execute / gripper wrappers (rclpy)
    edgehand/nodes/*.py        intent_node, vision_node, handover_node, pi_bridge (rclpy)
    launch/                    workstation.launch.py, jetson.launch.py
    test/test_core.py          pytest for the no-ROS modules (synthetic camera frames, fake robot)
    test/fake_cell.py          fake move_group / controllers / gripper / camera for testing the real nodes
    test/run_integration.sh    all nodes (real Gemma) against fake_cell: text, STOP, Pi buttons
  sim/handover.py              MuJoCo end-to-end test: overhead camera → vision → handover → operator tug
```

**Tested so far** (Oct 23, on a Mac with ROS 2 Humble from RoboStack):

| Test | Command (from `EdgeHand/ros2_ws/src/edgehand/`) | Result |
|---|---|---|
| Unit tests | `python -m pytest test/test_core.py` | 20/20 |
| Gemma 4 E4B intent, 7 phrasings ("cross-head", "the flat one", "star", a hammer, off-topic) | `python -m edgehand.intent <model> < phrases.txt` | 7/7 correct, ~1.4 s each |
| All nodes vs fake MoveIt: text → Gemma → vision → MoveIt → tug → release; STOP mid-move; Pi REQUEST + RELEASE over UDP | `bash test/run_integration.sh` | 3/3 |
| Pi watchdog | heartbeats stop | `/edgehand/stop` within 0.45 s, none while beating |
| MuJoCo, random tool poses: vision → pick → present → 12 N tug | from `EdgeHand/`: `../GemmaHackathon/sim/.venv/bin/python -m sim.handover --trials 20` (needs the sim venv and models from `GemmaHackathon/README.md` §10) | 20/20, vision ≤ 3 mm, handle towards the operator every time, release 0.1 s after the pull, no table contact |

**Not tested before the event:** `moveit_client.py` against a real `move_group` (only against `fake_cell.py`), MoveIt's own plans, the ros2_kortex controllers, a real camera, a real screwdriver's shaft and handle colours, and the real arm's joint-torque noise. Phase 3 step 1 (`use_fake_hardware:=true`) is the first place MoveIt and the nodes run together.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ping 192.168.1.10` fails | Cable, static IP on the RJ45, Wi-Fi off if it uses 192.168.1.x |
| Jetson topics missing on the workstation | Same `ROS_DOMAIN_ID`; try `fastdds_peers.xml` (Phase 1.3); `ros2 multicast receive` / `send` to test |
| Two drivers / controller_manager errors | Only the MoveIt launch should run; kill any `kortex_bringup` launch |
| `vision_node` NOT CALIBRATED | All 4 tags visible, lighting, print size 5 cm |
| Tool not detected | Handle colour outside its `TOOLS` HSV range: run `vision_node` with `show:=true` and tune it |
| Arm grips air / hits the tray | `TCP_OFFSET_M` (Phase 1.6) and `TABLE_Z` |
| Releases on its own | Raise `TUG_TORQUE_NM`, or wait longer before the baseline is taken (`SETTLE_S`) |
| Never releases on a tug | Lower `TUG_TORQUE_NM`; check `/joint_states` efforts aren't zero (they are with fake hardware) |
| Arm stops by itself | Pi watchdog: is `edgehand_pi.py` running, and is `pi_bridge` getting heartbeats? |
| MoveIt "no plan found" | Target outside reach or `WORKSPACE`; move the tray or `HANDOVER_XYZ` closer |

MIT for code written at the event. Gemma models are subject to the Gemma terms of use.
