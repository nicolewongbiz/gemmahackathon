"""MuJoCo end-to-end test of the EdgeHand logic, no ROS: overhead camera -> vision.py -> handover.py on a simulated
Gen3 + Robotiq 2F-85, with a simulated operator pulling the handle.

    ../GemmaHackathon/sim/.venv/bin/python -m sim.handover              (from EdgeHand/)
    ... -m sim.handover --trials 10                                     random tool poses, success rate
Needs the Menagerie models: bash ../GemmaHackathon/sim/get_assets.sh
"""
import argparse, sys
from pathlib import Path
import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ros2_ws" / "src" / "edgehand"))
from edgehand import config                                    # noqa: E402
from edgehand.handover import Handover                          # noqa: E402
from edgehand.vision import TableMap, detect_tools, draw        # noqa: E402

MENAGERIE = HERE.parent.parent / "GemmaHackathon" / "sim" / "menagerie"
OUT = HERE / "out"
DT, CTRL_EVERY, FPS = 0.002, 10, 30
JOINTS = [f"joint_{i}" for i in range(1, 8)]
SHAFT_L, HANDLE_L, HANDLE_R, SHAFT_R = 0.10, 0.10, 0.012, 0.003


def tag_png(tid):
    p = OUT / "tags" / f"tag{tid}.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tid, 8)
        m = cv2.resize(cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255), (400, 400),
                       interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(p), cv2.cvtColor(m, cv2.COLOR_GRAY2RGB))
    return str(p)


def build(tool_xy, tool_yaw):
    """tool_xy = ferrule position, tool_yaw = direction the handle points."""
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "kinova_gen3" / "gen3.xml"))
    spec.option.timestep = DT
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.attach(mujoco.MjSpec.from_file(str(MENAGERIE / "robotiq_2f85" / "2f85.xml")), prefix="g_",
                site=spec.site("pinch_site"))
    spec.site("pinch_site").name = "flange"
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1                       # the Gen3 controller cancels gravity
    spec.add_exclude(bodyname1="base_link", bodyname2="shoulder_link")
    wb = spec.worldbody
    for pos in ((0.4, -0.6, 1.6), (0.4, 0.6, 1.6), (-0.5, 0, 1.4)):
        wb.add_light(pos=pos, dir=-np.array(pos) / np.linalg.norm(pos), diffuse=[0.45] * 3)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.8, 0.8, 0.02], pos=[0.3, 0, -0.02],
                rgba=[0.05, 0.05, 0.06, 1], friction=[0.9, 0.005, 0.0001])
    half = config.TAG_SIZE_M * 10 / 8 / 2
    for tid, (x, y) in config.TABLE_TAGS.items():
        spec.add_texture(name=f"tag{tid}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=tag_png(tid))
        mat = spec.add_material(name=f"tag{tid}", specular=0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag{tid}"
        mesh = spec.add_mesh(name=f"tag{tid}", inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        mesh.uservert = [-half, -half, 0, half, -half, 0, half, half, 0, -half, half, 0]
        mesh.usertexcoord = [0, 1, 1, 1, 1, 0, 0, 0]
        mesh.userface = [0, 1, 2, 0, 2, 3]
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"tag{tid}", material=f"tag{tid}",
                    pos=[x, y, 0.0004], contype=0, conaffinity=0)
    # screwdriver: body frame at the ferrule, +x towards the handle
    tool = wb.add_body(name="tool", pos=[*tool_xy, HANDLE_R + 0.001],
                       quat=[np.cos(tool_yaw / 2), 0, 0, np.sin(tool_yaw / 2)])
    tool.add_freejoint()
    q_x = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]       # capsule/cylinder axis (z) -> body x
    tool.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[HANDLE_R, HANDLE_L / 2, 0],
                  pos=[HANDLE_L / 2, 0, 0], quat=q_x, rgba=[0.8, 0.1, 0.1, 1], mass=0.05, condim=6,
                  friction=[1.0, 0.01, 0.002])      # condim 6 = rolling resistance; without it a round handle rolls away
    tool.add_geom(name="shaft", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SHAFT_R, SHAFT_L / 2, 0],
                  pos=[-SHAFT_L / 2, 0, 0], quat=q_x, rgba=[0.72, 0.72, 0.75, 1], mass=0.02, condim=6,
                  friction=[1.0, 0.01, 0.002])
    tool.add_site(name="grasp", pos=[-config.GRASP_FROM_FERRULE_M, 0, 0])
    tool.add_site(name="handle_mid", pos=[HANDLE_L / 2, 0, 0])
    wb.add_camera(name="overhead", pos=[0.475, 0, 1.0], fovy=45)          # fixed, looking straight down
    wb.add_camera(name="side", pos=[1.3, -0.9, 0.9], xyaxes=[0.57, 0.82, 0, -0.33, 0.23, 0.92], fovy=50)
    return spec.compile()


class SimRobot:
    """The handover Robot interface on MuJoCo (flange poses, like MoveIt's end_effector_link)."""
    def __init__(self, m, video=False):
        self.m, self.d = m, mujoco.MjData(m)
        self.ik_d = mujoco.MjData(m)
        self.qa = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.da = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        self.act = np.array([m.actuator(j).id for j in JOINTS])
        self.grip = m.actuator("g_fingers_actuator").id
        self.lo = np.array([m.joint(j).range[0] if m.joint(j).limited else -np.inf for j in JOINTS])
        self.hi = np.array([m.joint(j).range[1] if m.joint(j).limited else np.inf for j in JOINTS])
        self.flange = m.site("flange").id
        self.q_home = m.key("home").qpos[self.qa].copy()
        tq = m.joint(m.body("tool").jntadr[0]).qposadr[0]
        mujoco.mj_resetDataKeyframe(m, self.d, m.key("home").id)
        self.d.qpos[tq:tq + 7] = m.qpos0[tq:tq + 7]
        self.q_park = np.array(config.PARK_JOINTS)
        self.d.qpos[self.qa] = self.q_park
        self.d.ctrl[self.act] = self.q_cmd = self.q_park.copy()
        self.pull = None                 # (start time, force N) of the simulated operator's tug
        self.released_at = None
        self.handle_towards_operator = None
        self.robot_bodies = {b for b in range(m.nbody) if self._under(b, m.body("shoulder_link").id)}
        self.events = set()
        self.time_limit = 90.0
        self.video = video
        self.frames = []
        self._vr = mujoco.Renderer(m, 540, 960) if video else None
        self.wait(1.0)                   # let the screwdriver settle on the table

    def _under(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    # ---- physics ----
    def wait(self, seconds):
        tool = self.m.body("tool").id
        if self.d.time > self.time_limit:
            raise TimeoutError(f"sim time limit {self.time_limit} s")
        for _ in range(int(seconds / DT)):
            self.d.xfrc_applied[tool] = 0
            if self.pull and self.d.time >= self.pull[0] and self.released_at is None:
                f = np.array([*config.OPERATOR_DIR[:2], 0.0]) * self.pull[1]
                self.d.xfrc_applied[tool, :3] = f         # operator pulls the handle towards themselves
            mujoco.mj_step(self.m, self.d)
            self._contacts()
            if self.video and int(self.d.time * FPS) != int((self.d.time - DT) * FPS):
                self._vr.update_scene(self.d, camera="side")
                self.frames.append(self._vr.render()[:, :, ::-1].copy())

    def _contacts(self):
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for mine, other in ((b1, c.geom2), (b2, c.geom1)):
                if mine in self.robot_bodies and self.m.geom(other).name == "table":
                    self.events.add(f"{self.m.body(mine).name} hit the table")

    # ---- kinematics ----
    def ik(self, p_t, R_t, q0, iters=100):
        d, q = self.ik_d, q0.copy()
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            d.qpos[self.qa] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p, R = d.site_xpos[self.flange], d.site_xmat[self.flange].reshape(3, 3)
            e = np.concatenate([p_t - p, 0.5 * sum(np.cross(R[:, i], R_t[:, i]) for i in range(3))])
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, d, jp, jr, self.flange)
            J = np.vstack([jp[:, self.da], jr[:, self.da]])
            A = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, e) + (np.eye(7) - J.T @ np.linalg.solve(A, J)) @ (0.05 * (self.q_home - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    def _track(self, qs):
        for q in qs:
            self.q_cmd = q
            self.d.ctrl[self.act] = q
            self.wait(DT * CTRL_EVERY)
        self.wait(0.3)

    def move_free(self, p, R):
        # MoveIt would plan in joint space; joint interpolation at a speed like VEL_SCALE = 0.2
        q1, ep, er = self.ik(np.asarray(p), R, self.q_cmd, iters=400)
        if ep > 0.003 or er > 0.02:
            return False
        n = int(max(np.abs(q1 - self.q_cmd)) / (0.2 * 1.4) / (DT * CTRL_EVERY)) + 1
        q0 = self.q_cmd.copy()
        self._track(q0 + (q1 - q0) * s for s in (0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)))
        return True

    def move_line(self, p, R, collisions=True):
        p0 = self.d.site_xpos[self.flange].copy()
        n = int(np.linalg.norm(np.asarray(p) - p0) / 0.05 / (DT * CTRL_EVERY)) + 1     # 5 cm/s
        qs, q = [], self.q_cmd
        for i in range(1, n + 1):
            q, ep, er = self.ik(p0 + (np.asarray(p) - p0) * i / n, R, q, iters=50)
            if ep > 0.003:
                return False
            qs.append(q)
        self._track(qs)
        return True

    def gripper(self, pos):
        self.d.ctrl[self.grip] = pos / config.GRIPPER_CLOSED * 255
        if pos == config.GRIPPER_OPEN and self.pull and self.d.time >= self.pull[0]:
            self.released_at = self.d.time
            # is the handle on the operator's side of where the gripper holds the shaft?
            g, hm = self.d.site_xpos[self.m.site("grasp").id], self.d.site_xpos[self.m.site("handle_mid").id]
            self.handle_towards_operator = float((hm - g)[:2] @ np.array(config.OPERATOR_DIR[:2])) > 0.03
        self.wait(1.0)
        return True

    def park(self):
        q0 = self.q_cmd.copy()
        n = int(max(np.abs(self.q_park - q0)) / 0.3 / (DT * CTRL_EVERY)) + 1
        self._track(q0 + (self.q_park - q0) * i / n for i in range(1, n + 1))
        return True

    def efforts(self):
        # measured joint torque ~ what the position servo pushes with (+ sensor noise)
        tau = self.d.qfrc_actuator[self.da] + np.random.normal(0, 0.05, 7)
        return dict(zip(JOINTS, tau))

    def cancel(self):
        pass

    def render(self, cam, h=720, w=1280):
        r = mujoco.Renderer(self.m, h, w)
        r.update_scene(self.d, camera=cam)
        return r.render()[:, :, ::-1].copy()


def episode(rng, video=False, pull_delay=3.0, pull_n=12.0):
    xs = [p[0] for p in config.TABLE_TAGS.values()]
    ys = [p[1] for p in config.TABLE_TAGS.values()]
    yaw = rng.uniform(-np.pi, np.pi)
    ferrule = (rng.uniform(min(xs) + 0.08, max(xs) - 0.08), rng.uniform(min(ys) + 0.08, max(ys) - 0.08))
    m = build(ferrule, yaw)
    r = SimRobot(m, video)
    # ---- vision on the rendered overhead frame ----
    frame = r.render("overhead")
    tm = TableMap()
    if not tm.update(frame):
        return {"ok": False, "msg": "not calibrated"}
    tools = detect_tools(frame, tm)
    cv2.imwrite(str(OUT / "overhead.jpg"), draw(frame, tm, tools))
    if not tools:
        return {"ok": False, "msg": "no tool detected"}
    t = tools[0]
    truth = r.d.site_xpos[m.site("grasp").id][:2]
    err = float(np.linalg.norm(t["grasp_xy"] - truth))
    # ---- handover with the operator pulling once the arm is holding ----
    h = Handover(r, log=lambda *a: None)
    orig_wait_for_tug = h._wait_for_tug

    def schedule_pull():
        r.pull = (r.d.time + pull_delay, pull_n)
        orig_wait_for_tug()
    h._wait_for_tug = schedule_pull
    try:
        ok, msg = h.run(t["grasp_xy"], t["handle_dir"])
    except TimeoutError as e:
        ok, msg = False, f"{e} (state: holding, last delta torque {h.detector.last_delta:.2f} N*m)"
    res = {"ok": ok and r.handle_towards_operator is True, "msg": msg, "vision_err_mm": err * 1000,
           "handle_towards_operator": r.handle_towards_operator, "events": sorted(r.events), "robot": r}
    if r.released_at is not None:
        res["release_after_pull_s"] = r.released_at - r.pull[0]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    if a.trials:
        n_ok = 0
        for i in range(a.trials):
            res = episode(rng)
            n_ok += res["ok"]
            print(f"trial {i + 1:2d}: {'OK  ' if res['ok'] else 'FAIL'} {res['msg']} | vision err "
                  f"{res.get('vision_err_mm', float('nan')):.1f} mm | released "
                  f"{res.get('release_after_pull_s', float('nan')):.2f} s after the pull | {res.get('events', [])}")
        print(f"success {n_ok}/{a.trials}")
    else:
        res = episode(rng, video=True)
        print({k: v for k, v in res.items() if k != "robot"})
        r = res.get("robot")
        if r and r.frames:
            h, w = r.frames[0].shape[:2]
            vw = cv2.VideoWriter(str(OUT / "handover.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            for f in r.frames:
                vw.write(f)
            vw.release()
            print("video:", OUT / "handover.mp4")
