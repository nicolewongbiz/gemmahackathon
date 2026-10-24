"""Run the ball -> box task in MuJoCo with the real task code (scripts/ball_to_box.py).
    sim/.venv/bin/python -m sim.run                        one episode, video in sim/out/episode.mp4
    sim/.venv/bin/python -m sim.run --trials 30            randomised scenes, success rate
    sim/.venv/bin/python -m sim.run --set Z_GRASP=0.06     override any config value
    sim/.venv/bin/python -m sim.run --trials 10 --sweep Z_GRASP=0.04,0.05,0.06
    sim/.venv/bin/python -m sim.run --trial 16             replay randomised trial 16 (same --seed) with video
"""
import argparse, ast, time
from collections import Counter
import cv2
import numpy as np
import config
from scripts import ball_to_box
from sim.world import OUT, SimArm, SimCamera, SimWorld, sample_scene


def episode(scene, video=False, quiet=False):
    w = SimWorld(scene, video=video, verbose=not quiet)
    arm, cam = SimArm(w), SimCamera(w)
    try:
        claimed, msg = ball_to_box.run(arm, cam, log=(lambda *a: None) if quiet else print)
    except ValueError as e:          # check_workspace refused a target
        claimed, msg = False, f"refused: {e}"
    return {"ok": w.ball_in_box(), "claimed": claimed, "msg": msg, "events": w.events, "world": w,
            "ball": w.ball_pos()}


def trials(n, seed):
    rng = np.random.default_rng(seed)
    res = []
    for i in range(n):
        r = episode(sample_scene(rng, randomize=True), quiet=True)
        res.append(r)
        print(f"  trial {i + 1:2d}: {'OK  ' if r['ok'] else 'FAIL'} {r['msg']}"
              + (f"  | {len(r['events'])} sim events" if r["events"] else ""))
    ok = sum(r["ok"] for r in res)
    print(f"success {ok}/{n} = {100 * ok / n:.0f}%")
    kinds = Counter(e[2].split(" (")[0] for r in res for e in r["events"])
    for k, c in kinds.most_common(8):
        print(f"  {c:3d}x {k}")
    return ok / n


def apply_set(pairs):
    for kv in pairs:
        k, v = kv.split("=", 1)
        assert hasattr(config, k), f"no config.{k}"
        setattr(config, k, ast.literal_eval(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--trial", type=int, help="replay this randomised trial with video")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="NAME=value config overrides")
    ap.add_argument("--sweep", help="NAME=v1,v2,... : run --trials for each value")
    a = ap.parse_args()
    config.DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    apply_set(a.set)
    t0 = time.time()
    if a.sweep:
        k, vals = a.sweep.split("=", 1)
        summary = []
        for v in vals.split(","):
            apply_set([f"{k}={v}"])
            print(f"== {k} = {v}")
            summary.append((v, trials(a.trials or 10, a.seed)))
        print("\n".join(f"{k}={v}: {100 * s:.0f}%" for v, s in summary))
    elif a.trials:
        trials(a.trials, a.seed)
    else:
        rng = np.random.default_rng(a.seed)
        scenes = [sample_scene(rng, randomize=True) for _ in range(a.trial or 0)]
        r = episode(scenes[-1] if scenes else sample_scene(), video=True)
        r["world"].save_video(OUT / "episode.mp4")
        cv2.imwrite(str(OUT / "wrist_detections.jpg"), cv2.imread(str(config.DATA / "last_ball.jpg")))
        print(f"\nresult: {'BALL IN BOX' if r['ok'] else 'FAILED'} | task says: {r['msg']} | "
              f"ball ended at {np.round(r['ball'], 3)}")
        for t, step, msg in r["events"]:
            print(f"  t={t:5.1f}s  {step}: {msg}")
        print(f"video: {OUT / 'episode.mp4'}")
    print(f"({time.time() - t0:.0f} s)")
