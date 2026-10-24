"""EdgeHand accuracy test in MuJoCo with real Gemma: a tray of screwdrivers and mechanical tools, an operator request
in plain words, and the full pipeline on a simulated Gen3 + 2F-85: GemmaIntent (words -> tool) -> wrist camera at the
look pose -> vision.find_tools + namer.GemmaNamer (Gemma names each numbered tool) -> handover.py. Scores each stage
separately and end to end.

    ../GemmaHackathon/sim/.venv/bin/python -m sim.tools                 intent battery + 30 tray trials  (from EdgeHand/)
    ... -m sim.tools --trials 100 --seed 3                              more trials
    ... -m sim.tools --namer colour                                     name tools by grip colour instead of Gemma
    ... -m sim.tools --no-intent                                        skip the phrase battery
    ... -m sim.tools --video                                            first trial -> sim/out/tools.mp4

Each tool = coloured grip + a metal working end. The robot grips the metal 2 cm from the grip and presents the grip
to the operator. Pliers, wrench and hammer are sim-only classes: they're added to config.TOOLS / TOOL_DESCRIPTIONS
here, not in config.py, because the real tray only has the three screwdrivers.
"""
import argparse, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ros2_ws" / "src" / "edgehand"))
from edgehand import config                                    # noqa: E402

# ---- tool classes: grip colour (render RGBA + OpenCV HSV range) ----
GRIPS = {
    "phillips": [0.80, 0.10, 0.10, 1],
    "flathead": [0.95, 0.80, 0.05, 1],
    "torx":     [0.10, 0.70, 0.20, 1],
    "pliers":   [0.10, 0.25, 0.85, 1],
    "wrench":   [0.55, 0.15, 0.75, 1],
    "hammer":   [0.95, 0.45, 0.05, 1],
}
config.TOOLS = dict(config.TOOLS, **{
    "pliers": [((100, 120, 60), (125, 255, 255))],     # blue grips
    "wrench": [((130, 90, 60), (160, 255, 255))],      # purple dip-coated handle
    "hammer": [((10, 150, 90), (18, 255, 255))],       # orange rubber grip
})
config.TOOL_DESCRIPTIONS = dict(config.TOOL_DESCRIPTIONS, pliers="pliers with blue grips",
                                wrench="adjustable wrench with a purple handle", hammer="hammer with an orange grip")
from edgehand import intent, namer                              # noqa: E402
from edgehand.geometry import look_pose                         # noqa: E402
from edgehand.handover import Handover                          # noqa: E402
from edgehand.vision import TableMap, detect_tools, draw, find_tools   # noqa: E402
from sim.handover import CTRL_EVERY, DT, FPS, JOINTS, MENAGERIE, OUT, SimRobot, tag_png   # noqa: E402

# the real prompt has synonyms for the screwdrivers only; the wrench is the one mechanical tool with common aliases
intent.SYSTEM = intent.SYSTEM.replace("Reply ONLY", 'Also "spanner", "crescent", "adjustable" = wrench.\nReply ONLY')
GEMMA = str(HERE.parent.parent / "GemmaHackathon" / "data" / "gemma-4-E4B-it.litertlm")
METAL = [0.72, 0.72, 0.75, 1]

# ---- operator requests: (text, expected tool or None). "absent" requests name a real class that isn't on the tray ----
REQUESTS = {
    "phillips": {"direct": ["hand me the phillips screwdriver", "phillips please", "can I get the phillips"],
                 "synonym": ["pass me the cross-head", "I need the plus screwdriver", "give me the crosshead driver"],
                 "task": ["this screw has a cross-shaped slot, what do I use?", "I need to undo a cross-head screw"]},
    "flathead": {"direct": ["hand me the flathead", "flathead screwdriver please", "get me the flathead"],
                 "synonym": ["pass the slotted screwdriver", "I need the flat blade one", "give me the straight screwdriver"],
                 "task": ["this screw has a single straight slot", "I need to pry open this paint can lid"]},
    "torx":     {"direct": ["torx please", "hand me the torx driver", "can you pass the torx"],
                 "synonym": ["give me the star screwdriver", "I need the star bit", "pass the six-lobe driver"],
                 "task": ["this screw has a six-point star in it", "the laptop screws are star shaped, what do I need?"]},
    "pliers":   {"direct": ["hand me the pliers", "pliers please", "can I get the pliers"],
                 "synonym": ["pass me the grips", "give me the needle-nose", "I need the side cutters"],
                 "task": ["I need to bend this wire", "I need to grip this small nut and hold it", "help me pull out this cotter pin"]},
    "wrench":   {"direct": ["hand me the wrench", "wrench please", "give me the adjustable wrench"],
                 "synonym": ["pass the spanner", "I need the crescent", "give me the adjustable"],
                 "task": ["I need to loosen this hex bolt", "this nut is stuck, I need to turn it", "tighten this pipe fitting"]},
    "hammer":   {"direct": ["hand me the hammer", "hammer please", "can I get the hammer"],
                 "synonym": ["pass the claw hammer", "give me the ball-peen", "I need the mallet"],
                 "task": ["I need to knock this nail in", "tap this pin through the hole", "this nail is sticking out"]},
}
NOT_TOOLS = ["hand me the tape measure", "can you get me a coffee", "pass the drill", "thanks, that's all",
             "where is the soldering iron?", "what time is it?", "give me the utility knife", "stop"]


# ---- scene ----
def tool_parts(cls, rng):
    """Geometry in the tool's body frame: origin = where grip meets metal, +x along the grip. Returns
    (axis height z0, grip length, metal length, parts) with parts = (kind, name, size, pos, rgba, mass)."""
    g = GRIPS[cls]
    if cls in ("phillips", "flathead", "torx"):
        hl, sl = rng.uniform(0.08, 0.11), rng.uniform(0.06, 0.12)          # stubby to long
        return 0.012, hl, sl, [("cyl", "grip", (0.012, hl / 2), (hl / 2, 0, 0), g, 0.05),
                               ("cyl", "shaft", (0.003, sl / 2), (-sl / 2, 0, 0), METAL, 0.02)]
    if cls == "pliers":
        return 0.008, 0.11, 0.084, [("box", "grip", (0.055, 0.013, 0.008), (0.055, 0, 0), g, 0.08),
                                    ("box", "pivot", (0.03, 0.009, 0.006), (-0.03, 0, 0), METAL, 0.10),
                                    ("box", "jaws", (0.012, 0.006, 0.005), (-0.072, 0, 0), METAL, 0.02)]
    if cls == "wrench":
        return 0.007, 0.10, 0.086, [("box", "grip", (0.05, 0.011, 0.007), (0.05, 0, 0), g, 0.06),
                                    ("box", "neck", (0.025, 0.008, 0.007), (-0.025, 0, 0), METAL, 0.08),
                                    ("box", "head", (0.018, 0.025, 0.007), (-0.068, 0, 0), METAL, 0.10)]
    return 0.015, 0.12, 0.108, [("cyl", "grip", (0.015, 0.06), (0.06, 0, 0), g, 0.12),               # hammer
                                ("cyl", "neck", (0.011, 0.04), (-0.04, 0, 0), METAL, 0.08),
                                ("box", "head", (0.014, 0.055, 0.014), (-0.094, 0, 0), METAL, 0.25)]


def footprint(t):
    """Table points covering the tool, for keeping tools apart."""
    _, hl, ml, parts = t["geom"]
    pts = [(x, 0) for x in np.arange(-ml, hl + 0.005, 0.01)]
    for kind, _, size, pos, _, _ in parts:
        if kind == "box" and size[1] > 0.02:                            # wide heads stick out sideways
            pts += [(pos[0], s * size[1]) for s in (-1, 1)]
    c, s = np.cos(t["yaw"]), np.sin(t["yaw"])
    return np.array([[t["xy"][0] + c * x - s * y, t["xy"][1] + s * x + c * y] for x, y in pts])


def grasp_xy(t):
    """Where the robot grips: GRASP_FROM_FERRULE_M along the metal from the grip (the tool body's origin)."""
    c, s = np.cos(t["yaw"]), np.sin(t["yaw"])
    return np.array(t["xy"]) - config.GRASP_FROM_FERRULE_M * np.array([c, s])


def sample_tray(rng, n_tools, gap=0.06):
    """n_tools different classes, laid flat at random, at least `gap` apart and clear of the tags. Each grasp point
    is inside the tag rectangle, like on the real tray: vision ignores anything further out than TRAY_MARGIN_M."""
    tags = np.array(list(config.TABLE_TAGS.values()))
    lo, hi = tags.min(axis=0), tags.max(axis=0)
    for _ in range(500):
        tools = []
        for cls in rng.choice(list(GRIPS), n_tools, replace=False):
            t = {"cls": str(cls), "name": f"tool_{cls}", "geom": tool_parts(cls, rng)}
            for _ in range(300):
                t["xy"] = rng.uniform((0.30, -0.27), (0.65, 0.27))
                t["yaw"] = rng.uniform(-np.pi, np.pi)
                fp = footprint(t)
                if (fp[:, 0].min() > 0.24 and fp[:, 0].max() < 0.72 and np.abs(fp[:, 1]).max() < 0.34
                        and np.all((lo < grasp_xy(t)) & (grasp_xy(t) < hi))
                        and min(np.linalg.norm(fp[:, None] - tags[None], axis=2).min(), 1) > 0.065
                        and all(np.linalg.norm(fp[:, None] - footprint(o)[None], axis=2).min() > gap for o in tools)):
                    tools.append(t)
                    break
            else:
                break
        if len(tools) == n_tools:
            return tools
    raise ValueError(f"can't fit {n_tools} tools on the tray")


def build(tools):
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
    q_x = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]       # cylinder axis (z) -> body x
    for t in tools:
        z0, hl, _, parts = t["geom"]
        body = wb.add_body(name=t["name"], pos=[*t["xy"], z0 + 0.001],
                           quat=[np.cos(t["yaw"] / 2), 0, 0, np.sin(t["yaw"] / 2)])
        body.add_freejoint()
        for kind, part, size, pos, rgba, mass in parts:
            body.add_geom(name=f"{t['name']}_{part}", pos=pos, rgba=rgba, mass=mass, condim=6,
                          friction=[1.0, 0.01, 0.002],      # condim 6 = rolling resistance (round grips roll)
                          **({"type": mujoco.mjtGeom.mjGEOM_CYLINDER, "size": [*size, 0], "quat": q_x}
                             if kind == "cyl" else {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": size}))
        body.add_site(name=f"{t['name']}_grasp", pos=[-config.GRASP_FROM_FERRULE_M, 0, 0])
        body.add_site(name=f"{t['name']}_handle", pos=[hl / 2, 0, 0])
    wb.add_camera(name="overhead", pos=[0.475, 0, 1.0], fovy=45)
    wb.add_camera(name="side", pos=[1.3, -0.9, 0.9], xyaxes=[0.57, 0.82, 0, -0.33, 0.23, 0.92], fovy=50)
    return spec.compile()


class TrayRobot(SimRobot):
    """sim.handover.SimRobot with a tray of tools: the operator pulls whichever tool the arm is holding."""
    def __init__(self, m, tools, video=False):
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
        mujoco.mj_resetDataKeyframe(m, self.d, m.key("home").id)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                a = m.jnt_qposadr[j]
                self.d.qpos[a:a + 7] = m.qpos0[a:a + 7]
        q_look, ep, _ = self.ik(*look_pose(), self.q_home, iters=400)     # start where the arm waits
        assert ep < 0.003, f"look pose unreachable (IK off by {ep * 1000:.0f} mm)"
        self.d.qpos[self.qa] = q_look
        self.d.ctrl[self.act] = self.q_cmd = q_look.copy()
        self.tools = [t["name"] for t in tools]
        self.tool_ids = {m.body(n).id: n for n in self.tools}
        self.pull = None                 # (start time, force N) of the simulated operator's tug
        self.pulled = None               # the tool the operator is pulling
        self.released_at = self.handed = self.handle_towards_operator = None
        self.robot_bodies = {b for b in range(m.nbody) if self._under(b, m.body("shoulder_link").id)}
        self.pads = {m.geom(n).id for n in ("g_left_pad1", "g_left_pad2", "g_right_pad1", "g_right_pad2")}
        self.events = set()
        self.time_limit = 90.0
        self.video = video
        self.frames = []
        self._vr = mujoco.Renderer(m, 540, 960) if video else None
        self.wait(1.0)                   # let the tools settle on the table

    def held(self):
        """The tool lifted well off the table, if any."""
        up = [n for n in self.tools if self.d.site_xpos[self.m.site(f"{n}_grasp").id][2] > 0.15]
        return up[0] if up else None

    def wait(self, seconds):
        if self.d.time > self.time_limit:
            raise TimeoutError(f"sim time limit {self.time_limit} s")
        for _ in range(int(seconds / DT)):
            self.d.xfrc_applied[:] = 0
            if self.pull and self.d.time >= self.pull[0] and self.released_at is None:
                self.pulled = self.pulled or self.held()
                if self.pulled:          # empty gripper: there's nothing for the operator to pull
                    self.d.xfrc_applied[self.m.body(self.pulled).id, :3] = \
                        np.array([*config.OPERATOR_DIR[:2], 0.0]) * self.pull[1]
            mujoco.mj_step(self.m, self.d)
            self._contacts()
            if self.video and int(self.d.time * FPS) != int((self.d.time - DT) * FPS):
                self._vr.update_scene(self.d, camera="side")
                self.frames.append(self._vr.render()[:, :, ::-1].copy())

    def _contacts(self):
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for mine, g_mine, other in ((b1, c.geom1, c.geom2), (b2, c.geom2, c.geom1)):
                if mine not in self.robot_bodies:
                    continue
                ob = self.m.geom_bodyid[other]
                if self.m.geom(other).name == "table":
                    self.events.add(f"{self.m.body(mine).name} hit the table")
                elif ob in self.tool_ids and g_mine not in self.pads:
                    self.events.add(f"{self.m.body(mine).name} hit {self.tool_ids[ob]}")

    def gripper(self, pos):
        self.d.ctrl[self.grip] = pos / config.GRIPPER_CLOSED * 255
        if pos == config.GRIPPER_OPEN and self.pull and self.d.time >= self.pull[0]:
            self.released_at = self.d.time
            name = self.held()
            if name:
                self.handed = name.removeprefix("tool_")
                g = self.d.site_xpos[self.m.site(f"{name}_grasp").id]
                h = self.d.site_xpos[self.m.site(f"{name}_handle").id]
                self.handle_towards_operator = float((h - g)[:2] @ np.array(config.OPERATOR_DIR[:2])) > 0.03
        self.wait(1.0)
        return True


# ---- one trial ----
def trial(rng, gemma, n_tools, video=False, request=None, tool_namer=None):
    """request = (text, tool) forces that request, with the tool on the tray. tool_namer = namer.GemmaNamer, or None
    to name tools by grip colour (vision.detect_tools)."""
    tools = sample_tray(rng, n_tools)
    while request and request[1] not in [t["cls"] for t in tools]:
        tools = sample_tray(rng, n_tools)
    on_tray = [t["cls"] for t in tools]
    kind = "present" if request else rng.choice(["present", "absent", "none"], p=[0.75, 0.125, 0.125])
    if kind == "absent" and len(on_tray) == len(GRIPS):
        kind = "present"
    if request:
        want = request[1]
    elif kind == "present":
        want = str(rng.choice(on_tray))
    elif kind == "absent":
        want = str(rng.choice([c for c in GRIPS if c not in on_tray]))
    if kind == "none":
        text, parse_want, hand_want = str(rng.choice(NOT_TOOLS)), None, None
    else:
        style = str(rng.choice(list(REQUESTS[want])))
        text, parse_want = request[0] if request else str(rng.choice(REQUESTS[want][style])), want
        hand_want = want if kind == "present" else None
    res = {"text": text, "kind": kind, "on_tray": on_tray, "parse_want": parse_want, "hand_want": hand_want}

    t0 = time.time()
    parsed = gemma.parse(text)
    res.update(parsed=parsed["tool"], reply=parsed["reply"], gemma_s=time.time() - t0)

    m = build(tools)
    r = TrayRobot(m, tools, video)
    frame = r.render("wrist")                                  # the arm starts at the look pose
    tm = TableMap()
    if not tm.update(frame):
        return dict(res, handed=None, stage="vision", msg="not calibrated", robot=r)
    if tool_namer:
        found = find_tools(frame, tm)
        t0 = time.time()
        names = tool_namer.name(frame, found)
        res["namer_s"] = time.time() - t0
        dets = [dict(f, tool=n) for f, n in zip(found, names) if n]
    else:
        dets = detect_tools(frame, tm)
    # what vision got right: every tool found, with the right class, grasp point and handle direction
    truth = {t["cls"]: (r.d.site_xpos[m.site(f"{t['name']}_grasp").id][:2],
                        r.d.site_xpos[m.site(f"{t['name']}_handle").id][:2]) for t in tools}
    res["vision"] = {}
    for cls, (g, h) in truth.items():
        d = [x for x in dets if x["tool"] == cls]
        ok = bool(d) and np.linalg.norm(d[0]["grasp_xy"] - g) < 0.015 and \
            float(d[0]["handle_dir"] @ ((h - g) / np.linalg.norm(h - g))) > 0.9
        res["vision"][cls] = ok
    res["false_dets"] = [x["tool"] for x in dets if x["tool"] not in on_tray]
    res["frame"] = draw(frame, tm, dets)

    if parsed["tool"] is None:
        return dict(res, handed=None, stage="intent: no tool", msg=parsed["reply"], robot=r)
    d = [x for x in dets if x["tool"] == parsed["tool"]]
    if not d:
        return dict(res, handed=None, stage="vision: not on tray", msg=f"no {parsed['tool']} found", robot=r)
    h = Handover(r, log=lambda *a: None)
    orig = h._wait_for_tug

    def schedule_pull():
        r.pull = (r.d.time + 3.0, 12.0)
        orig()
    h._wait_for_tug = schedule_pull
    try:
        ok, msg = h.run(d[0]["grasp_xy"], d[0]["handle_dir"])
    except TimeoutError as e:
        ok, msg = False, f"{e} (holding {r.held() or 'nothing'}, tug delta {h.detector.last_delta:.2f} N*m)"
    return dict(res, handed=r.handed, handle_ok=r.handle_towards_operator, stage="handover", msg=msg,
                events=sorted(r.events), robot=r)


def correct(res):
    if res["handed"] != res["hand_want"]:
        return False
    return res["handed"] is None or res.get("handle_ok") is True


# ---- reports ----
def intent_battery(gemma):
    print("== Gemma intent battery: every phrase once")
    by_cat, wrong = defaultdict(lambda: [0, 0]), []
    items = [(t, cls, style) for cls, styles in REQUESTS.items() for style, ts in styles.items() for t in ts]
    items += [(t, None, "not a tool") for t in NOT_TOOLS]
    times = []
    for text, want, style in items:
        t0 = time.time()
        got = gemma.parse(text)["tool"]
        times.append(time.time() - t0)
        for key in (style, f"{want or 'none'}"):
            by_cat[key][0] += got == want
            by_cat[key][1] += 1
        if got != want:
            wrong.append(f"    \"{text}\" -> {got} (want {want})")
    ok = sum(by_cat[s][0] for s in ("direct", "synonym", "task", "not a tool"))
    print(f"overall {ok}/{len(items)} = {100 * ok / len(items):.0f}%   ({np.mean(times):.1f} s per request)")
    for k in ("direct", "synonym", "task", "not a tool", *GRIPS):
        c, n = by_cat[k]
        print(f"  {k:11} {c:2d}/{n:<2d} {100 * c / n:4.0f}%")
    if wrong:
        print("  misses:\n" + "\n".join(wrong))


def report(results):
    n = len(results)
    ok = [correct(r) for r in results]
    print(f"\n== End to end: {sum(ok)}/{n} = {100 * sum(ok) / n:.0f}% correct "
          f"(right tool handed over handle-first, or correctly nothing)")
    for kind in ("present", "absent", "none"):
        rs = [r for r in results if r["kind"] == kind]
        if rs:
            c = sum(correct(r) for r in rs)
            label = {"present": "asked for a tool on the tray", "absent": "asked for a tool not on the tray",
                     "none": "not a tool request"}[kind]
            print(f"  {label:34} {c:3d}/{len(rs):<3d} {100 * c / len(rs):4.0f}%")
    gi = sum(r["parsed"] == r["parse_want"] for r in results)
    print(f"stage 1  Gemma intent        {gi:3d}/{n:<3d} {100 * gi / n:4.0f}%")
    vis = [v for r in results for v in r.get("vision", {}).values()]
    fd = sum(len(r.get("false_dets", [])) for r in results)
    print(f"stage 2  vision (per tool)   {sum(vis):3d}/{len(vis):<3d} {100 * sum(vis) / max(len(vis), 1):4.0f}%"
          f"   + {fd} detections of a class not on the tray")
    tried = [r for r in results if r["stage"] == "handover"]
    got = [r for r in tried if r["handed"] == r["parsed"] and r.get("handle_ok")]
    print(f"stage 3  grasp + handover    {len(got):3d}/{len(tried):<3d} {100 * len(got) / max(len(tried), 1):4.0f}%"
          f"   (of the times the arm went for a tool)")
    per = defaultdict(lambda: [0, 0])
    for r in results:
        if r["kind"] == "present":
            per[r["hand_want"]][0] += correct(r)
            per[r["hand_want"]][1] += 1
    print("by tool (asked for, on the tray): " + "  ".join(f"{k} {c}/{t}" for k, (c, t) in sorted(per.items())))
    vis_cls = defaultdict(lambda: [0, 0])
    for r in results:
        for k, v in r.get("vision", {}).items():
            vis_cls[k][0] += v
            vis_cls[k][1] += 1
    print("vision by tool:                   " + "  ".join(f"{k} {c}/{t}" for k, (c, t) in sorted(vis_cls.items())))
    ev = Counter(e for r in results for e in r.get("events", []))
    if ev:
        print("sim events: " + ", ".join(f"{k} x{c}" for k, c in ev.most_common(6)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--tools", type=int, default=4, help="tools on the tray per trial")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-intent", action="store_true", help="skip the phrase battery")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--namer", choices=["gemma", "colour"], default="gemma", help="who names the tools it sees")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t_start = time.time()
    tool_namer = namer.GemmaNamer(GEMMA) if a.namer == "gemma" else None
    if tool_namer:      # one engine for both: a second copy of the model would compete for the GPU
        gemma = intent.GemmaIntent.__new__(intent.GemmaIntent)
        gemma.engine, gemma.system = tool_namer.engine, intent.SYSTEM.format(tools=", ".join(config.TOOLS))
    else:
        gemma = intent.GemmaIntent(GEMMA)
    print("tools:", ", ".join(config.TOOLS), "| tools named by", a.namer)
    if not a.no_intent:
        intent_battery(gemma)
    rng = np.random.default_rng(a.seed)
    results = []
    print(f"\n== {a.trials} tray trials, {a.tools} tools each")
    for i in range(a.trials):
        r = trial(rng, gemma, a.tools, video=a.video and i == 0, tool_namer=tool_namer)
        results.append(r)
        mark = "OK  " if correct(r) else "FAIL"
        print(f"{i + 1:3d} {mark} \"{r['text']}\" -> Gemma: {r['parsed']} | handed: {r['handed']} "
              f"(want {r['hand_want']}) | tray: {','.join(r['on_tray'])} | {r['stage']}: {r['msg']}", flush=True)
        if not correct(r) and "frame" in r:
            cv2.imwrite(str(OUT / f"tools_fail_{i + 1}.jpg"), r["frame"])
        rob = r.pop("robot")
        if a.video and i == 0 and rob.frames:
            h, w = rob.frames[0].shape[:2]
            vw = cv2.VideoWriter(str(OUT / "tools.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            for f in rob.frames:
                vw.write(f)
            vw.release()
        r.pop("frame", None)
    report(results)
    print(f"({time.time() - t_start:.0f} s)")
