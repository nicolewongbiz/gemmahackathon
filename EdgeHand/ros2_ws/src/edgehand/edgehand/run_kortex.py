"""EdgeHand without ROS: wrist camera -> vision -> Gemma names the tools -> handover.py on the Kortex API.
Runs on the computer the camera is cabled to (this Mac). Red E-stop in hand; MoveIt isn't involved.

    python -m edgehand.run_kortex --air                 first runs: every grasp 3 cm above the tools
    python -m edgehand.run_kortex                       type requests ("hand me the green screwdriver"); Ctrl-C stops
    python -m edgehand.run_kortex --tool torx           one request, no intent model
    python -m edgehand.run_kortex --look-only           move to the look pose, show what vision sees, no pick
    python -m edgehand.run_kortex --namer colour        name tools by handle colour instead of Gemma
    python -m edgehand.run_kortex --namer first --tool x  ONE tool on the tray: whatever is found is the requested one

Refuses to pick on config.py's placeholder calibration: run measure_tags first (data/calibration.json).
"""
import argparse, json, sys, tempfile, threading, time
from pathlib import Path
import cv2
import numpy as np
from edgehand import config
from edgehand.camera import IR_HINT, StreamCamera, is_infrared
from edgehand.geometry import table_z
from edgehand.handover import Handover
from edgehand.vision import TableMap, detect_tools, draw, find_tools


def calibration_gate(allow_defaults):
    f = getattr(config, "CALIBRATION_FILE", None)
    if f is None:
        print("NO SITE CALIBRATION: config.py placeholders (TABLE_TAGS, flat table at TABLE_Z).")
        if not allow_defaults:
            print("Run `python -m edgehand.measure_tags` on this cell first, or --allow-defaults to look (no picks).")
            sys.exit(2)
    else:
        meta = json.loads(Path(f).read_text())
        print(f"site calibration {f}: site '{meta.get('site', '?')}', measured {meta.get('measured', '?')}")
    if not config.TABLE_TOUCH_Z:
        print(f"TABLE_TOUCH_Z is empty: table assumed flat at z = {config.TABLE_Z} (README Phase 1.6)")
    else:
        zs = {i: round(table_z(*config.TABLE_TAGS[i]), 4) for i in config.TABLE_TOUCH_Z}   # raises if implausible
        print(f"table height at the tags (m): {zs}")
    print(f"clearance {config.TABLE_CLEARANCE_M * 1000:.0f} mm, fingertips {config.FINGERTIP_CLOSED_M:.3f} m, "
          f"air gap {config.AIR_GAP_M * 1000:.0f} mm, empty-gripper threshold {config.GRIPPER_EMPTY_POS}, "
          f"segmentation {config.SEGMENTATION}")


class Eyes:
    """Fresh frames from the wrist camera, and what vision makes of them."""
    def __init__(self, namer):
        self.n = 0
        self.cam = StreamCamera(on_frame=self._count)
        self.tm = TableMap()
        self.namer = namer
        self.want = None                       # the requested class, for --namer first
        self.warned_ir = False
        self.last = None

    def _count(self, f):
        self.n += 1

    def frame(self):
        """A frame taken after now (the stream lags a few frames)."""
        target, t0 = self.n + config.LOCATE_SKIP_FRAMES, time.time()
        while self.n < target and time.time() - t0 < 5.0:
            time.sleep(0.02)
        return self.cam.frame

    def see(self, show):
        """[(tool dict with 'tool' name)], or None when not calibrated."""
        f = self.frame()
        if f is not None and is_infrared(f) and not self.warned_ir:
            self.warned_ir = True                  # geometry still works in IR (tags, contrast); colour naming won't
            print("CAMERA:", IR_HINT, "Carrying on in IR: use --namer first or gemma, not colour.")
        if f is None or not self.tm.update(f):
            print("NOT CALIBRATED:", getattr(self.tm, "why_not", lambda: "tags not found")())
            return None
        if self.namer == "first":
            tools = [dict(t, tool=self.want or "?") for t in find_tools(f, self.tm)]
            if len(tools) > 1:
                print(f"--namer first: {len(tools)} tools found, taking the biggest; put ONE tool on the tray")
                tools = tools[:1]
        elif self.namer is None:
            tools = detect_tools(f, self.tm)
        else:
            found = find_tools(f, self.tm)
            names = self.namer.name(f, found)
            tools = [dict(t, tool=n) for t, n in zip(found, names) if n]
        print("vision:", ", ".join(f"{t['tool']} at ({t['grasp_xy'][0]:.3f}, {t['grasp_xy'][1]:.3f})" for t in tools)
              or "no tools")
        self.last = draw(f, self.tm, tools)
        cv2.imwrite(str(Path(tempfile.gettempdir()) / "edgehand_last_look.jpg"), self.last)   # for debugging
        if show:
            cv2.imshow("edgehand", self.last)
            cv2.waitKey(1)
        return tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--air", action="store_true", help="AIR_GAP_M = 0.03: every grasp 3 cm above the tools")
    ap.add_argument("--tool", help="one request for this tool class, no intent model")
    ap.add_argument("--look-only", action="store_true")
    ap.add_argument("--namer", choices=["gemma", "colour", "first"], default=config.VISION_NAMER)
    ap.add_argument("--show", action="store_true", help="window with the detections")
    ap.add_argument("--allow-defaults", action="store_true")
    ap.add_argument("--robot", choices=["kortex", "none"], default="kortex", help="none = vision only")
    a = ap.parse_args()
    if a.air:
        config.AIR_GAP_M = 0.03
    calibration_gate(a.allow_defaults)

    namer = "first" if a.namer == "first" else None
    if a.namer == "gemma":
        from edgehand.namer import GemmaNamer
        namer = GemmaNamer()
    intent = None
    if not a.tool and not a.look_only:
        from edgehand.intent import GemmaIntent
        intent = GemmaIntent() if isinstance(namer, (str, type(None))) else _share_engine(namer)
    eyes = Eyes(namer)

    robot = None
    if a.robot == "kortex":
        from edgehand.kortex_robot import KortexRobot
        robot = KortexRobot()
    h = Handover(robot, log=print, state=lambda s: print("state:", s)) if robot else None

    def relocate(tool):
        time.sleep(config.LOOK_SETTLE_S)
        eyes.want = tool
        tools = eyes.see(a.show) or []
        hit = [t for t in tools if t["tool"] == tool]
        return (hit[0]["grasp_xy"], hit[0]["handle_dir"]) if hit else None

    def on_interrupt():
        if h:
            h.stop()

    try:
        if robot:
            print("moving to the look pose")
            if not robot.look():
                print("can't reach the look pose; stopping")
                return
        time.sleep(config.LOOK_SETTLE_S)
        if a.look_only or not robot:
            while True:
                eyes.see(True)
                if cv2.waitKey(500) == 27:
                    break
            return
        while True:
            if a.tool:
                tool = a.tool
            else:
                text = input("\nrequest> ").strip()
                if not text:
                    continue
                r = intent.parse(text)
                print("gemma:", r["reply"])
                tool = r["tool"]
                if tool is None:
                    continue
            if tool not in config.TOOLS:
                print(f"unknown tool class {tool!r}; classes: {list(config.TOOLS)}")
                if a.tool:
                    return
                continue
            found = relocate(tool)
            if found is None:
                print(f"can't see the {tool}")
                if a.tool:
                    return
                continue
            ok, msg = h.run(found[0], found[1], relocate=lambda: relocate(tool))
            print("handover:", "OK" if ok else "FAILED", msg)
            if a.tool:
                return
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping the arm")
        on_interrupt()
    finally:
        if robot:
            robot.close()


def _share_engine(namer):
    """One Gemma engine for naming and intent (a second copy would compete for the GPU)."""
    from edgehand import intent
    g = intent.GemmaIntent.__new__(intent.GemmaIntent)
    g.engine, g.system = namer.engine, intent.SYSTEM.format(tools=", ".join(config.TOOLS))
    return g


if __name__ == "__main__":
    main()
