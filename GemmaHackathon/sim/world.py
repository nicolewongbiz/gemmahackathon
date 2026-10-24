"""MuJoCo stand-in for the real cell: Kinova Gen3 + Robotiq 2F-85, black table, AprilTags, white ball, blue box.

SimArm and SimCamera have the same methods as robot.arm.KinovaArm and perception.camera.KinovaVisionCamera,
so the real pick_and_place / TableMap / perception code runs against them unchanged.

Kortex conventions copied here, so poses that work in sim mean the same thing on the arm:
  - base frame: x forward, y left, z up, origin at the base mount (= table surface)
  - move_to(x, y, z) targets the Kortex TOOL frame: the flange pushed SIM_TOOL_FRAME_Z along its z axis
  - orientation is TOOL_ORIENT as Kortex theta_x/y/z: extrinsic XYZ, R = Rz @ Ry @ Rx (checked against the
    Gen3 Home pose, which reads theta = (90, 0, 90) both here and in the Web App)
  - reach_pose moves the tool in a straight line at MAX_LINEAR_SPEED
"""
from pathlib import Path
import cv2
import mujoco
import numpy as np
import config

HERE = Path(__file__).parent
MENAGERIE = HERE / "menagerie"
OUT = HERE / "out"
DT = 0.002              # physics step (s)
CTRL_EVERY = 10         # IK / control update every N physics steps (50 Hz)
VIDEO_FPS = 30
ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]


def kortex_R(tx, ty, tz):
    """Kortex theta_x, theta_y, theta_z (deg) -> rotation matrix."""
    x, y, z = np.radians([tx, ty, tz])
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def kortex_theta(R):
    return tuple(np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)),
                             np.arctan2(R[1, 0], R[0, 0])]))


def tag_texture(tid):
    """tag36h11 with its 1-cell white quiet zone, same as tags/make_tags.py prints."""
    path = OUT / "tags" / f"tag{tid}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tid, 8)
        img = cv2.copyMakeBorder(marker, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return path


def add_tag(spec, wb, tid, x, y, z):
    half = config.TAG_SIZE_M * 10 / 8 / 2           # black square + 1-cell quiet zone on each side
    spec.add_texture(name=f"tag{tid}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tag_texture(tid)))
    mat = spec.add_material(name=f"tag{tid}", specular=0, shininess=0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag{tid}"
    mesh = spec.add_mesh(name=f"tag{tid}")
    mesh.uservert = [-half, -half, 0, half, -half, 0, half, half, 0, -half, half, 0]
    mesh.usertexcoord = [0, 1, 1, 1, 1, 0, 0, 0]
    mesh.userface = [0, 1, 2, 0, 2, 3]
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    # Tag image "up" points along -x (towards the arm) — detection doesn't care about rotation
    wb.add_geom(name=f"tag{tid}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"tag{tid}", material=f"tag{tid}",
                pos=[x, y, z], quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], contype=0, conaffinity=0)


def add_tray(wb, name, xy, yaw, size, rgba):
    """Open-top box: floor + 4 walls. size = outer (x, y, wall height)."""
    bx, by, bh = size[0] / 2, size[1] / 2, size[2]
    t = 0.004
    body = wb.add_body(name=name, pos=[*xy, 0], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    for part, sz, pos in (("floor", [bx, by, t / 2], [0, 0, t / 2]),
                          ("wall_px", [t / 2, by, bh / 2], [bx - t / 2, 0, bh / 2]),
                          ("wall_nx", [t / 2, by, bh / 2], [-bx + t / 2, 0, bh / 2]),
                          ("wall_py", [bx, t / 2, bh / 2], [0, by - t / 2, bh / 2]),
                          ("wall_ny", [bx, t / 2, bh / 2], [0, -by + t / 2, bh / 2])):
        body.add_geom(name=f"{name}_{part}", type=mujoco.mjtGeom.mjGEOM_BOX, size=sz, pos=pos, rgba=rgba)


def build_model(scene):
    """scene: dict from sample_scene(). Returns a compiled MjModel."""
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "kinova_gen3" / "gen3.xml"))
    grip = mujoco.MjSpec.from_file(str(MENAGERIE / "robotiq_2f85" / "2f85.xml"))
    spec.option.timestep = DT
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.attach(grip, prefix="g_", site=spec.site("pinch_site"))
    spec.site("pinch_site").name = "flange"
    spec.visual.global_.offwidth = max(config.FRAME_W, 1280)
    spec.visual.global_.offheight = max(config.FRAME_H, 720)
    spec.visual.headlight.ambient = [0.25] * 3
    spec.visual.headlight.diffuse = [0.3] * 3
    spec.visual.quality.shadowsize = 4096
    # Kinova Vision colour stream: vertical FOV ~42 deg; the render size sets the aspect ratio
    spec.camera("wrist").fovy = scene["cam_fovy"]

    wb = spec.worldbody
    for pos, dif in (((0.4, -0.6, 1.6), 0.5), ((0.4, 0.6, 1.6), 0.4), ((-0.5, 0, 1.4), 0.3)):
        light = wb.add_light(pos=pos, dir=-np.array(pos) / np.linalg.norm(pos))
        light.diffuse = [dif * scene["light"]] * 3
        light.castshadow = True

    black = spec.add_material(name="black_table", rgba=[0.03, 0.03, 0.03, 1], specular=0.05, shininess=0.1)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.8, 0.8, 0.02], pos=[0.3, 0, -0.02],
                material="black_table", friction=[0.8, 0.005, 0.0001])

    # AprilTags lying flat where they really are (scene["tag_xy"] = config + tape-measure error)
    for tid, (x, y) in scene["tag_xy"].items():
        add_tag(spec, wb, tid, x, y, 0.0004)

    # White ball (ball -> box task)
    if "ball_xy" in scene:
        r = scene["ball_r"]
        ball = wb.add_body(name="ball", pos=[*scene["ball_xy"], r + 0.0005])
        ball.add_freejoint()
        ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], mass=scene["ball_mass"],
                      rgba=[0.93, 0.93, 0.93, 1], condim=6, friction=[scene["ball_friction"], 0.005, 0.0005],
                      solref=[0.004, 1])

    # Open-top blue box (ball -> box task)
    if "box_xy" in scene:
        add_tray(wb, "box", scene["box_xy"], scene["box_yaw"], config.SIM_BOX_SIZE, [0.08, 0.28, 0.85, 1])

    # Sorting task: grey bins with their tag (10 = A, 11 = B) on the floor, where pick_and_place drops
    tag_ids = {v: k for k, v in config.BIN_TAGS.items()}
    for name, (x, y) in scene.get("bins", {}).items():
        add_tray(wb, f"bin_{name}", (x, y), 0.0, config.SIM_BIN_SIZE, [0.55, 0.55, 0.58, 1])
        add_tag(spec, wb, tag_ids[name], x, y, 0.0044)

    # Sorting task objects: {"name", "type": "sphere" | "box", "size", "rgba", "mass", "pos"}
    for o in scene.get("objects", []):
        b = wb.add_body(name=o["name"], pos=o["pos"])
        b.add_freejoint()
        gtype = mujoco.mjtGeom.mjGEOM_SPHERE if o["type"] == "sphere" else mujoco.mjtGeom.mjGEOM_BOX
        b.add_geom(name=o["name"], type=gtype, size=o["size"], rgba=o["rgba"], mass=o["mass"], condim=4,
                   friction=[0.9, 0.005, 0.0005])

    # Overview camera for videos (front-right, looking at the work area)
    wb.add_camera(name="overview", pos=[1.35, -0.85, 0.85], xyaxes=[0.53, 0.85, 0, -0.32, 0.2, 0.93], fovy=50)

    # The Gen3's controller cancels gravity; without this the sim arm sags ~5 mm and grasp heights shift
    loose = {"world", "ball", "box"} | {f"bin_{n}" for n in scene.get("bins", {})} | \
            {o["name"] for o in scene.get("objects", [])}
    for b in spec.bodies:
        if b.name not in loose:
            b.gravcomp = 1
    spec.add_exclude(bodyname1="base_link", bodyname2="world")
    # base_link is welded to the world, so MuJoCo's parent/child filter doesn't cover it: without this the
    # base and shoulder meshes touch and joint 1 never turns
    spec.add_exclude(bodyname1="base_link", bodyname2="shoulder_link")
    return spec.compile()


def sample_scene(rng=None, randomize=False):
    """Where things are. randomize=True = domain randomisation for success-rate trials."""
    rng = rng or np.random.default_rng()
    xs = [p[0] for p in config.TABLE_TAGS.values()]
    ys = [p[1] for p in config.TABLE_TAGS.values()]
    s = {"tag_xy": dict(config.TABLE_TAGS), "ball_r": config.SIM_BALL_RADIUS, "ball_mass": config.SIM_BALL_MASS,
         "ball_friction": 0.8, "box_xy": config.SIM_BOX_XY, "box_yaw": 0.0, "light": 1.0, "cam_fovy": 41.84,
         "ball_xy": ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)}
    if randomize:
        m = 0.05   # keep the ball off the tags
        s["ball_xy"] = (rng.uniform(min(xs) + m, max(xs) - m), rng.uniform(min(ys) + m, max(ys) - m))
        e = config.SIM_TAG_ERROR_M
        s["tag_xy"] = {i: (x + rng.normal(0, e), y + rng.normal(0, e)) for i, (x, y) in config.TABLE_TAGS.items()}
        s["ball_r"] *= rng.uniform(0.9, 1.1)
        s["ball_mass"] *= rng.uniform(0.7, 1.5)
        s["ball_friction"] = rng.uniform(0.4, 1.0)
        s["box_xy"] = (config.SIM_BOX_XY[0] + rng.uniform(-0.03, 0.03), config.SIM_BOX_XY[1] + rng.uniform(-0.03, 0.03))
        s["box_yaw"] = rng.uniform(-0.3, 0.3)
        s["light"] = rng.uniform(0.6, 1.3)
        s["cam_fovy"] = 41.84 + rng.uniform(-1.5, 1.5)
    return s


class SimWorld:
    def __init__(self, scene=None, video=False, verbose=True):
        self.scene = scene or sample_scene()
        self.m = build_model(self.scene)
        self.d = mujoco.MjData(self.m)
        self.ik_d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([self.m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo = np.array([self.m.joint(j).range[0] if self.m.joint(j).limited else -np.inf for j in ARM_JOINTS])
        self.hi = np.array([self.m.joint(j).range[1] if self.m.joint(j).limited else np.inf for j in ARM_JOINTS])
        self.flange = self.m.site("flange").id
        self.grip_act = self.m.actuator("g_fingers_actuator").id
        self.arm_act = np.array([self.m.actuator(j).id for j in ARM_JOINTS])
        home = self.m.key("home")
        self.q_home = home.qpos[self.qadr].copy()
        mujoco.mj_resetDataKeyframe(self.m, self.d, home.id)
        # the keyframe only covers the arm; put the ball / objects back where the scene says
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                a = self.m.jnt_qposadr[j]
                self.d.qpos[a:a + 7] = self.m.qpos0[a:a + 7]
        self.d.ctrl[self.arm_act] = self.q_home
        mujoco.mj_forward(self.m, self.d)
        self.robot_bodies = self._subtree("shoulder_link")
        self.pad_geoms = {self.m.geom(n).id for n in ("g_left_pad1", "g_left_pad2", "g_right_pad1", "g_right_pad2")}
        trays = (["box"] if "box_xy" in self.scene else []) + [f"bin_{n}" for n in self.scene.get("bins", {})]
        tray_ids = [0] + [self.m.body(n).id for n in trays]
        self.env_geoms = {i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] in tray_ids}
        loose = (["ball"] if "ball_xy" in self.scene else []) + [o["name"] for o in self.scene.get("objects", [])]
        self.obj_geoms = {self.m.geom(n).id for n in loose}
        self.events = []                  # (time, step, message): collisions and IK problems
        self._seen = set()
        self.verbose = verbose
        self.step_name = ""
        self.video = video
        self.frames = []
        self._vid_r = mujoco.Renderer(self.m, 540, 960) if video else None
        self._cam_r = None
        self.settle(0.5)

    def _subtree(self, root):
        rid = self.m.body(root).id
        out = set()
        for b in range(self.m.nbody):
            p = b
            while p > 0:
                if p == rid:
                    out.add(b)
                    break
                p = self.m.body_parentid[p]
        return out

    # ---- kinematics ----
    def tool_pose(self, d=None):
        d = d or self.d
        R = d.site_xmat[self.flange].reshape(3, 3).copy()
        return d.site_xpos[self.flange] + R[:, 2] * config.SIM_TOOL_FRAME_Z, R

    def ik(self, p_t, R_t, q0, iters=200, tol=1e-4):
        """Damped least squares on the Kortex tool frame, with a pull towards Home in the null space."""
        d = self.ik_d
        q = q0.copy()
        bid = self.m.site_bodyid[self.flange]
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p, R = self.tool_pose(d)
            e_p = p_t - p
            e_r = 0.5 * (np.cross(R[:, 0], R_t[:, 0]) + np.cross(R[:, 1], R_t[:, 1]) + np.cross(R[:, 2], R_t[:, 2]))
            e = np.concatenate([e_p, e_r])
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 10 * tol:
                break
            mujoco.mj_jac(self.m, d, jp, jr, p, bid)
            J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.05 * (self.q_home - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q, float(np.linalg.norm(e_p)), float(np.degrees(np.linalg.norm(e_r)))

    # ---- simulation ----
    def _physics(self, n):
        for i in range(n):
            mujoco.mj_step(self.m, self.d)
            self._check_contacts()
            if self.video and int(self.d.time * VIDEO_FPS) != int((self.d.time - DT) * VIDEO_FPS):
                self._vid_r.update_scene(self.d, camera="overview")
                self.frames.append(self._vid_r.render()[:, :, ::-1].copy())

    def settle(self, seconds):
        self._physics(int(seconds / DT))

    def _check_contacts(self):
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            b1, b2 = self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]
            robot = (b1 in self.robot_bodies) or (b2 in self.robot_bodies)
            if not robot:
                continue
            other = g2 if b1 in self.robot_bodies else g1
            mine = g1 if other == g2 else g2
            if other in self.env_geoms:
                what = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, other) or "table"
            elif other in self.obj_geoms and mine not in self.pad_geoms:
                what = f"{self.m.geom(other).name} (not with the finger pads)"
            else:
                continue
            link = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[mine])
            msg = f"{link.replace('left_', '').replace('right_', '')} hit {what}"
            if (self.step_name, msg) not in self._seen:      # once per step, not every physics tick
                self._seen.add((self.step_name, msg))
                self.events.append((round(self.d.time, 2), self.step_name, msg))
                if self.verbose:
                    print(f"  [sim] {self.step_name}: {msg}")

    def body_pos(self, name):
        return self.d.xpos[self.m.body(name).id].copy()

    def place(self, name, xyz, yaw=0.0):
        """Teleport a free body (a person putting an object down) and let it come to rest."""
        a = self.m.joint(self.m.body(name).jntadr[0]).qposadr[0]
        v = self.m.joint(self.m.body(name).jntadr[0]).dofadr[0]
        self.d.qpos[a:a + 7] = [*xyz, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.qvel[v:v + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def ball_pos(self):
        return self.d.xpos[self.m.body("ball").id].copy()

    def ball_in_box(self):
        b = self.ball_pos()
        bid = self.m.body("box").id
        local = self.d.xmat[bid].reshape(3, 3).T @ (b - self.d.xpos[bid])
        hx, hy, h = config.SIM_BOX_SIZE[0] / 2, config.SIM_BOX_SIZE[1] / 2, config.SIM_BOX_SIZE[2]
        return abs(local[0]) < hx and abs(local[1]) < hy and local[2] < h

    def render(self, camera, h, w):
        if self._cam_r is None or (self._cam_r.height, self._cam_r.width) != (h, w):
            self._cam_r = mujoco.Renderer(self.m, h, w)
        self._cam_r.update_scene(self.d, camera=camera)
        return self._cam_r.render()[:, :, ::-1].copy()

    def save_video(self, path):
        if not self.frames:
            return
        h, w = self.frames[0].shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
        for f in self.frames:
            vw.write(f)
        vw.release()


class SimArm:
    """Drop-in for KinovaArm."""
    def __init__(self, world):
        self.w = world
        self.q_cmd = world.q_home.copy()

    def move_to(self, x, y, z):
        from robot.arm import check_workspace
        check_workspace(x, y, z)
        w = self.w
        w.step_name = f"move_to({x:.3f}, {y:.3f}, {z:.3f})"
        p0, R0 = w.tool_pose()
        p1, R1 = np.array([x, y, z + config.Z_TABLE_OFFSET]), kortex_R(*config.TOOL_ORIENT)
        # Kortex reach_pose: straight-line tool path, speed-limited in translation and rotation
        ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        T = max(np.linalg.norm(p1 - p0) / config.MAX_LINEAR_SPEED, ang / config.MAX_ANGULAR_SPEED, 0.2)
        n = max(1, int(T / (DT * CTRL_EVERY)))
        rel = R0.T @ R1
        axis_ang = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        worst = (0.0, 0.0)
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)    # smooth start/stop
            Rs = R0 @ _axis_angle_R(axis_ang, np.radians(ang) * s)
            q, ep, er = w.ik(p0 + (p1 - p0) * s, Rs, self.q_cmd, iters=50)
            worst = max(worst, (ep, er))
            self.q_cmd = q
            w.d.ctrl[w.arm_act] = q
            w._physics(CTRL_EVERY)
        if worst[0] > 0.005 or worst[1] > 2:
            w.events.append((round(w.d.time, 2), w.step_name, f"unreachable: IK off by {worst[0] * 1000:.0f} mm / {worst[1]:.0f} deg"))
            raise ValueError(f"Kinova aborted the move (sim: {w.step_name} unreachable, "
                             f"IK off by {worst[0] * 1000:.0f} mm / {worst[1]:.0f} deg)")
        w.settle(0.3)

    def gripper(self, v):
        self.w.step_name = f"gripper({v:.2f})"
        self.w.d.ctrl[self.w.grip_act] = float(np.clip(v, 0, 1)) * 255
        self.w.settle(1.0)    # KinovaArm.gripper sleeps 1 s

    def look(self):
        self.move_to(*config.LOOK_POSE)
        self.w.settle(0.3)

    def home(self):
        w = self.w
        w.step_name = "home"
        q0 = self.q_cmd.copy()
        n = int(max(np.abs(w.q_home - q0)) / np.radians(config.MAX_ANGULAR_SPEED) / (DT * CTRL_EVERY)) + 1
        for i in range(1, n + 1):
            self.q_cmd = q0 + (w.q_home - q0) * i / n
            w.d.ctrl[w.arm_act] = self.q_cmd
            w._physics(CTRL_EVERY)
        w.settle(0.5)

    def stop(self):
        pass

    def close(self):
        pass


class SimCamera:
    """Drop-in for KinovaVisionCamera: renders the Gen3 wrist camera."""
    def __init__(self, world):
        self.w = world

    def read(self):
        return self.w.render("wrist", config.FRAME_H, config.FRAME_W)

    def close(self):
        pass


def _axis_angle_R(v, angle):
    n = np.linalg.norm(v)
    if n < 1e-9 or abs(angle) < 1e-9:
        return np.eye(3)
    k = v / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
