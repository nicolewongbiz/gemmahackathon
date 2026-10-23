# Kinova + calibration

## 1. Kortex API

The Kortex Python API is **not on PyPI**. Download the wheel (e.g. `kortex_api-2.6.0.post3-py3-none-any.whl`) from Kinova's Artifactory (linked from the [Kinova-kortex2_Gen3_G3L](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L) README), then:
```bash
pip install kortex_api-*.whl
```
Kortex pins an old `protobuf`; if imports fail, `pip install "protobuf==3.20.*"`.

Confirm the model: **Gen3** (7-DoF, Robotiq 2F-85) or **Gen3 lite** (6-DoF, own gripper). Set `ARM_MODEL` in `config.py`.

## 2. Network

- Arm default IP: `192.168.1.10`, user/pass `admin`/`admin`
- Laptop interface to the arm: static `192.168.1.11/24`
- Check: open `http://192.168.1.10` (Kinova Web App) — jog the arm and read tool poses here.

```bash
python -m scripts.test_kinova      # connects, prints pose, gripper open/close, +5 cm up and back
```
**Keep the E-stop in hand the first time.**

## 3. AprilTags

```bash
python -m scripts.print_tags       # writes data/tags.pdf: tag36h11 IDs 0,1,2,3,10,11 at 5 cm
```
Print at 100% scale. Measure a printed tag: the black square must be 5.0 cm.

| ID | Placement |
|---|---|
| 0 | Table corner nearest the arm, right |
| 1 | Table corner nearest the arm, left |
| 2 | Far left |
| 3 | Far right |
| 10 | On bin A |
| 11 | On bin B |

Tape them flat. Keep them out of the arm's path.

## 4. Measure tag positions in the robot base frame

Option A (fast, ~1 cm): tape measure from the base centre: x forward, y left (Kinova base frame). Enter centres in `TABLE_TAGS`.

Option B (accurate, ~2 mm): in the Web App, jog the gripper tip to touch each tag centre, read `x, y` from the tool pose, enter in `TABLE_TAGS`. Also record the table height `z` → `Z_TABLE_OFFSET`.

## 5. Verify the homography

```bash
python -m perception.calibrate
```
- All four corner IDs must show and the status must read CALIBRATED.
- Click on a tag centre → the logged metres must match `TABLE_TAGS` within ~1 cm.
- Click a random spot, then jog the arm there in the Web App to confirm.

The homography maps the table **plane** only; tall objects' centroids shift outward from the camera centre. Mount the camera as high and as close to vertical as possible.

## 6. Tune grasps

```bash
python -m scripts.test_click_pick
```
Click an object → arm picks it → drops in bin A.

| Setting | Start | Tune if… |
|---|---|---|
| `Z_HOVER` | 0.15 m | arm hits things on approach → raise |
| `Z_GRASP` | 0.025 m | misses → lower in 5 mm steps; hits table → raise |
| `TOOL_ORIENT` | (180, 0, 90) | gripper not pointing straight down → read from Web App while pointing down |
| `GRIPPER_CLOSED` | 0.7 | drops objects → raise; crushes → lower |
| `MAX_LINEAR_SPEED` | 0.15 m/s | keep slow near judges |

Use objects 3–7 cm wide and soft (foam blocks, plush fruit, sponges, small plastic tools).

## 7. Safety

- Workspace bounds = rectangle of tags 0–3 + `WORKSPACE_MARGIN_M`; targets outside are rejected before sending.
- The arm returns home between picks (clears the camera view and makes motion predictable).
- `stop` intent → `base.Stop()` immediately; physical E-stop always within reach of a teammate.
