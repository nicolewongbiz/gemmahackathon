"""Show Me Once in MuJoCo with the real Gemma planner. A simulated person sorts a few objects into the bins (teach),
Gemma infers the rule from the wrist-camera keyframes, new objects land on the table, and main.py's sort loop runs
with Gemma choosing every pick. Gemma's choices are scored separately from the arm's physics.

    sim/.venv/bin/python -m sim.sort                     colour rule (red -> A, blue -> B), video in sim/out/sort.mp4
    sim/.venv/bin/python -m sim.sort --rule shape        balls -> A, blocks -> B
    sim/.venv/bin/python -m sim.sort --correct           then "actually, red things go in B" and a second batch
    sim/.venv/bin/python -m sim.sort --trials 3          every rule x 3 random layouts, no video
    --objects 6     objects per batch      --labels     the eyes node's Gemma names each mark (as on the Jetson)
    --seed N        layout                 --set NAME=value   override any config value
"""
import argparse, time
import cv2
import numpy as np
import config
from perception import eyes as eyes_mod
from perception.objects import draw_marks
from planner import brain as brain_mod
from robot.arm import pick_and_place
from sim.run import apply_set
from sim.world import OUT, VIDEO_FPS, SimArm, SimCamera, SimWorld

COLOURS = {"red": [0.85, 0.12, 0.1, 1], "blue": [0.1, 0.3, 0.9, 1]}
SHAPES = ("ball", "block")
# The hidden rule the simulated person sorts by. The demo varies both colour and shape, so only one rule fits it.
RULES = {
    "colour": lambda o: "A" if o["colour"] == "red" else "B",
    "shape": lambda o: "A" if o["shape"] == "ball" else "B",
}
# What the user says to correct the rule, and the rule it should become
CORRECTIONS = {
    "colour": ("actually, red things go in B", lambda o: "B" if o["colour"] == "red" else "A"),
    "shape": ("actually, balls go in B", lambda o: "B" if o["shape"] == "ball" else "A"),
}
DEMO = [("red", "ball"), ("blue", "block"), ("red", "block"), ("blue", "ball")]


def obj_spec(name, colour, shape, pos):
    s = config.SIM_OBJ_SIZE / 2
    return {"name": name, "colour": colour, "shape": shape, "type": "sphere" if shape == "ball" else "box",
            "size": [s, 0, 0] if shape == "ball" else [s * 0.9] * 3, "rgba": COLOURS[colour], "mass": 0.02,
            "pos": pos}


def random_batch(rng, n):
    """n (colour, shape) pairs with both colours and both shapes present."""
    while True:
        b = [(str(rng.choice(list(COLOURS))), str(rng.choice(SHAPES))) for _ in range(n)]
        if {c for c, _ in b} == set(COLOURS) and {s for _, s in b} == set(SHAPES):
            return b


def scatter(rng, n, min_gap=0.08, margin=0.06):
    """n table positions inside the tag rectangle, at least min_gap apart (the finder merges touching blobs)."""
    xs = [p[0] for p in config.TABLE_TAGS.values()]
    ys = [p[1] for p in config.TABLE_TAGS.values()]
    for _ in range(1000):
        pts = []
        while len(pts) < n:
            for _ in range(200):
                p = np.array([rng.uniform(min(xs) + margin, max(xs) - margin),
                              rng.uniform(min(ys) + margin, max(ys) - margin)])
                if all(np.linalg.norm(p - q) >= min_gap for q in pts):
                    pts.append(p)
                    break
            else:
                break
        if len(pts) == n:
            return pts
    raise ValueError(f"can't fit {n} objects {min_gap} m apart on the table")


def staging(i):
    """Behind the arm, out of the wrist camera's view: where objects wait until the 'person' puts them down."""
    return [-0.30 - 0.07 * (i // 10), -0.32 + 0.07 * (i % 10), config.SIM_OBJ_SIZE / 2 + 0.001]


class StreamVideo:
    """Writes SimWorld's video frames straight to disk; a sort episode is minutes long."""
    def __init__(self, path):
        self.path, self.vw = path, None

    def append(self, f):
        if self.vw is None:
            self.vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, f.shape[1::-1])
        self.vw.write(f)

    def close(self):
        if self.vw is not None:
            self.vw.release()


class Episode:
    def __init__(self, rule, rng, n_objects=6, correct=False, labels=False, video=False, log=print):
        self.rule, self.rng, self.labels, self.log = rule, rng, labels, log
        self.batches = [random_batch(rng, n_objects)] + ([random_batch(rng, n_objects)] if correct else [])
        self.objs = [obj_spec(f"demo{i}", c, s, None) for i, (c, s) in enumerate(DEMO)]
        for b, batch in enumerate(self.batches):
            self.objs += [obj_spec(f"b{b}_{i}", c, s, None) for i, (c, s) in enumerate(batch)]
        for i, o in enumerate(self.objs):
            o["pos"] = staging(i)
        self.by_name = {o["name"]: o for o in self.objs}
        scene = {"tag_xy": dict(config.TABLE_TAGS), "light": 1.0, "cam_fovy": 41.84,
                 "bins": dict(config.SIM_BIN_XY), "objects": self.objs}
        # the sim knows where the bins really are; BIN_DROP is only used when a bin tag is covered or out of view
        config.BIN_DROP = {b: (*xy, 0.15) for b, xy in config.SIM_BIN_XY.items()}
        self.w = SimWorld(scene, video=video, verbose=False)
        if video:
            self.w.frames = StreamVideo(OUT / "sort.mp4")
        self.arm, self.cam = SimArm(self.w), SimCamera(self.w)
        eyes_mod.make_camera = lambda: self.cam
        self.eyes = eyes_mod.LocalEyes()
        brain_mod.RULE_FILE = OUT / "rule.json"          # don't overwrite the real data/rule.json
        self.brain = brain_mod.Brain()
        self.brain.rule = {"rule": "unknown", "bins": {"A": "?", "B": "?"}}
        if labels:   # the Jetson's E2B on Ollama isn't here; the laptop model stands in
            config.EYES_GEMMA_BACKEND, config.EYES_GEMMA_MODEL = "litert", config.LITERT_MODEL

    # ---- the simulated person ----
    def put_on_table(self, names):
        for name, p in zip(names, scatter(self.rng, len(names))):
            self.w.place(name, (*p, config.SIM_OBJ_SIZE / 2 + 0.002), self.rng.uniform(-np.pi, np.pi))
        self.w.settle(0.5)

    def where(self, name):
        """'A' / 'B' if inside that bin, 'table' if on the table, else 'lost'."""
        p = self.w.body_pos(name)
        hx, hy, h = config.SIM_BIN_SIZE[0] / 2, config.SIM_BIN_SIZE[1] / 2, config.SIM_BIN_SIZE[2]
        for b, (bx, by) in config.SIM_BIN_XY.items():
            if abs(p[0] - bx) < hx and abs(p[1] - by) < hy and p[2] < h:
                return b
        return "table" if p[2] < 0.05 and p[0] > 0.1 else "lost"

    def describe(self, name):
        o = self.by_name[name]
        return f"{o['colour']} {o['shape']}"

    # ---- the real flow from main.py ----
    def teach(self):
        self.arm.look()
        self.eyes.snapshot_empty()
        frames = []

        def keyframe():      # = LocalEyes.teach_stop: the settled frame with BIN A / BIN B drawn on it
            self.w.settle(0.5)
            f = self.cam.read()
            self.eyes.tmap.update(f)
            frames.append(draw_marks(f, {}, self.eyes.tmap.bins_px))

        demo = [o["name"] for o in self.objs if o["name"].startswith("demo")]
        self.put_on_table(demo)
        keyframe()
        corners = {"A": [(-0.05, -0.05), (0.05, 0.05)], "B": [(-0.05, -0.05), (0.05, 0.05)]}   # clear of the bin tag
        for name in demo:
            b = RULES[self.rule](self.by_name[name])
            dx, dy = corners[b].pop(0)
            bx, by = config.SIM_BIN_XY[b]
            self.w.place(name, (bx + dx, by + dy, 0.04))
            keyframe()
        for i, f in enumerate(frames):
            cv2.imwrite(str(OUT / f"teach_{i}.jpg"), f)
        t0 = time.time()
        said = self.brain.teach(frames)
        self.log(f"teach: {len(frames)} keyframes -> Gemma ({time.time() - t0:.1f} s)")
        self.log(f"  rule:  {self.brain.rule['rule']}")
        self.log(f"  bins:  A = {self.brain.rule['bins'].get('A')} | B = {self.brain.rule['bins'].get('B')}")
        self.log(f"  says:  \"{said}\"")

    def sort(self, b, truth):
        names = [o["name"] for o in self.objs if o["name"].startswith(f"b{b}_")]
        self.put_on_table(names)
        self.log(f"on the table: {', '.join(self.describe(n) for n in names)}")
        steps = []
        for n in range(12):                                   # same cap as main.run_sort
            self.arm.look()
            s = self.eyes.scene(label=self.labels)
            if not s["calibrated"]:
                self.log("  can't see all four table tags"); break
            objs = s["objects"]
            if not objs:
                self.log("  table clear"); break
            cv2.imwrite(str(OUT / f"sort{b}_step{n + 1}.jpg"), s["marked"])
            t0 = time.time()
            step = self.brain.plan(s["marked"], list(objs), s["labels"])
            dt = time.time() - t0
            if step.get("object") is None:
                self.log(f"  step {n + 1}: Gemma says nothing fits: {step.get('say')}  ({dt:.1f} s)"); break
            # which real object is under that mark?
            xy = np.array(objs[step["object"]])
            near = [m for m in names if np.linalg.norm(self.w.body_pos(m)[:2] - xy) < 0.04]
            want = truth(self.by_name[near[0]]) if near else None
            ok = step.get("bin") == want
            steps.append({"ok": ok, "name": near[0] if near else None, "bin": step.get("bin"), "want": want})
            seen = "".join(k for k in ("A", "B") if k in self.eyes.tmap.bins_px) or "none"
            self.log(f"  step {n + 1}: Gemma -> #{step['object']} \"{step.get('label', '?')}\" to bin {step.get('bin')}"
                     f" | really {self.describe(near[0]) if near else 'nothing there'} -> {want}"
                     f"  {'OK' if ok else 'WRONG'}  ({dt:.1f} s, bin tags seen: {seen})")
            if step.get("bin") not in s["bins"]:
                self.log(f"  bin {step.get('bin')!r} isn't A or B"); break
            try:
                pick_and_place(self.arm, xy, s["bins"][step["bin"]])
            except ValueError as e:
                self.log(f"  arm: {e}"); break
        self.arm.home()
        final = {n: self.where(n) for n in names}
        return {"steps": steps, "final": final, "truth": {n: truth(self.by_name[n]) for n in names}}

    def run(self):
        t0 = time.time()
        self.teach()
        results = [self.sort(0, RULES[self.rule])]
        if len(self.batches) > 1:
            text, new_truth = CORRECTIONS[self.rule]
            said = self.brain.correct(text)
            self.log(f"correct: \"{text}\" -> A = {self.brain.rule['bins'].get('A')} | "
                     f"B = {self.brain.rule['bins'].get('B')} | says \"{said}\"")
            for n, where in results[0]["final"].items():       # clear what's left of batch 1
                if where == "table":
                    self.w.place(n, staging(self.objs.index(self.by_name[n])))
            results.append(self.sort(1, new_truth))
        if isinstance(self.w.frames, StreamVideo):
            self.w.frames.close()
        return {"rule": dict(self.brain.rule), "batches": results, "events": self.w.events,
                "seconds": time.time() - t0}


def summarise(res, log=print):
    for i, b in enumerate(res["batches"]):
        st, fin, tr = b["steps"], b["final"], b["truth"]
        right = sum(fin[n] == tr[n] for n in fin)
        wrong = sum(fin[n] in ("A", "B") and fin[n] != tr[n] for n in fin)
        left = sum(fin[n] == "table" for n in fin)
        lost = sum(fin[n] == "lost" for n in fin)
        # Gemma chose right but the object isn't in that bin: the arm's fault, not Gemma's
        fumbled = sum(s["ok"] and s["name"] and fin[s["name"]] != s["bin"] for s in st)
        log(f"batch {i + 1}{' (after correction)' if i else ''}: Gemma chose right {sum(s['ok'] for s in st)}/{len(st)}"
            f" | end result {right}/{len(fin)} in the right bin, {wrong} wrong bin, {left} left on table, {lost} lost"
            f" | arm fumbled {fumbled}")
    arm_hits = sorted({e[2] for e in res["events"]})
    if arm_hits:
        log("sim events: " + "; ".join(arm_hits[:6]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=list(RULES), default="colour")
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="NAME=value config overrides")
    a = ap.parse_args()
    config.DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    apply_set(a.set)
    rng = np.random.default_rng(a.seed)
    if a.trials:
        rows = []
        for rule in RULES:
            for i in range(a.trials):
                print(f"\n== {rule} rule, trial {i + 1}")
                r = Episode(rule, rng, a.objects, a.correct, a.labels).run()
                summarise(r)
                rows.append((rule, r))
        print("\nrule    | Gemma chose right | in the right bin")
        for rule in RULES:
            rs = [r for k, r in rows if k == rule]
            steps = [s for r in rs for b in r["batches"] for s in b["steps"]]
            fins = [(b["final"][n], b["truth"][n]) for r in rs for b in r["batches"] for n in b["final"]]
            print(f"{rule:7} | {sum(s['ok'] for s in steps):3d}/{len(steps):<3d}           "
                  f"| {sum(f == t for f, t in fins):3d}/{len(fins)}")
    else:
        r = Episode(a.rule, rng, a.objects, a.correct, a.labels, video=True).run()
        print()
        summarise(r)
        print(f"video: {OUT / 'sort.mp4'} | Gemma's images: {OUT}/teach_*.jpg, sort*_step*.jpg "
              f"({r['seconds']:.0f} s)")
