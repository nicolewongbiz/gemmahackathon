"""Validate, clamp and execute the actions Gemma returns.

Action schema (what Gemma must output inside "actions"):
  {"type": "move",    "dx": m, "dy": m, "dz": m}           relative translation, base frame
  {"type": "move_to", "x": m, "y": m, "z": m}              absolute position (any subset)
  {"type": "rotate",  "droll": deg, "dpitch": deg, "dyaw": deg}   relative tool rotation
  {"type": "gripper", "value": 0..1}                       0 = open, 1 = closed
  {"type": "home"}
  {"type": "stop"}
  {"type": "wait",    "seconds": s}
"""
import logging
import threading
import time

import config

log = logging.getLogger("commands")


class SafetyError(Exception):
    pass


def _num(a, key, default=0.0):
    v = a.get(key, default)
    if v is None:
        return default
    return float(v)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _check_inside(target, current):
    """Each axis must end inside WORKSPACE, or, if the arm already starts outside on that axis
    (e.g. it was jogged there from the Web App), at least not move further out, so it can recover."""
    ws = config.WORKSPACE
    for name, val, now in zip("xyz", target, current):
        lo, hi = ws[name]
        if lo <= val <= hi:
            continue
        if not lo <= now <= hi and max(lo - val, val - hi) <= max(lo - now, now - hi) + 1e-6:
            continue
        where = (f"The arm is already at {name}={now:.3f} m, outside the box; say 'go home' first. "
                 if not lo <= now <= hi else "")
        raise SafetyError(f"Target {name}={val:.3f} m is outside the workspace [{lo}, {hi}]. {where}"
                          "Edit WORKSPACE in config.py if this is wrong.")


def validate(actions):
    """Return a cleaned list of actions, or raise SafetyError."""
    if not isinstance(actions, list):
        raise SafetyError("actions must be a list")
    if len(actions) > config.MAX_ACTIONS:
        raise SafetyError(f"Too many actions ({len(actions)} > {config.MAX_ACTIONS})")
    out = []
    for a in actions:
        if not isinstance(a, dict) or "type" not in a:
            raise SafetyError(f"Bad action: {a}")
        t = str(a["type"]).lower()
        if t == "move":
            d = [_num(a, "dx"), _num(a, "dy"), _num(a, "dz")]
            if max(abs(v) for v in d) > config.MAX_STEP_M:
                raise SafetyError(f"Step too large ({max(abs(v) for v in d)*100:.0f} cm > "
                                  f"{config.MAX_STEP_M*100:.0f} cm). Ask for smaller moves.")
            out.append({"type": "move", "dx": d[0], "dy": d[1], "dz": d[2]})
        elif t == "move_to":
            out.append({"type": "move_to", **{k: float(a[k]) for k in ("x", "y", "z") if a.get(k) is not None}})
        elif t == "rotate":
            r = [_num(a, "droll"), _num(a, "dpitch"), _num(a, "dyaw")]
            if max(abs(v) for v in r) > config.MAX_ROT_DEG:
                raise SafetyError(f"Rotation too large (> {config.MAX_ROT_DEG} deg)")
            out.append({"type": "rotate", "droll": r[0], "dpitch": r[1], "dyaw": r[2]})
        elif t == "gripper":
            v = a.get("value")
            if isinstance(v, str):
                v = {"open": 0.0, "close": 1.0, "closed": 1.0}.get(v.lower(), 0.5)
            out.append({"type": "gripper", "value": _clamp(float(v if v is not None else 0.0), 0.0, 1.0)})
        elif t in ("home", "stop"):
            out.append({"type": t})
        elif t == "wait":
            out.append({"type": "wait", "seconds": _clamp(_num(a, "seconds", 1.0), 0.0, 10.0)})
        else:
            raise SafetyError(f"Unknown action type '{t}'")
    return out


def describe(a):
    t = a["type"]
    if t == "move":
        parts = []
        for k, pos, neg in (("dx", "forward", "back"), ("dy", "left", "right"), ("dz", "up", "down")):
            if abs(a[k]) > 1e-6:
                parts.append(f"{abs(a[k])*100:.1f} cm {pos if a[k] > 0 else neg}")
        return "move " + (", ".join(parts) or "0 cm")
    if t == "move_to":
        return "move to " + ", ".join(f"{k}={v:.3f} m" for k, v in a.items() if k != "type")
    if t == "rotate":
        return f"rotate roll {a['droll']:+.0f}°, pitch {a['dpitch']:+.0f}°, yaw {a['dyaw']:+.0f}°"
    if t == "gripper":
        return f"gripper {'open' if a['value'] < 0.05 else 'closed' if a['value'] > 0.95 else round(a['value'], 2)}"
    if t == "wait":
        return f"wait {a['seconds']:.1f} s"
    return t


class Executor:
    """Runs action lists one at a time in a background thread, so STOP always works."""

    def __init__(self, arm):
        self.arm = arm
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.busy = False
        self.log = []          # recent lines for the UI

    def _note(self, msg):
        log.info(msg)
        self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        self.log = self.log[-200:]

    def stop(self):
        self.stop_flag.set()
        try:
            self.arm.stop()
        finally:
            self._note("STOP")

    def submit(self, actions):
        if self.busy:
            raise SafetyError("Arm is busy. Wait or press STOP.")
        actions = validate(actions)
        self.stop_flag.clear()
        threading.Thread(target=self._run, args=(actions,), daemon=True).start()
        return actions

    def _run(self, actions):
        with self.lock:
            self.busy = True
            try:
                for a in actions:
                    if self.stop_flag.is_set():
                        self._note("aborted remaining actions")
                        break
                    self._note("→ " + describe(a))
                    self._one(a)
                self._note("done")
            except Exception as e:  # noqa: BLE001
                if "SESSION_NOT_IN_CONTROL" in str(e):
                    e = ("Another client has control of the arm (Kinova Web App or gamepad). "
                         "Close the Web App tab / put the gamepad down, then restart run_real.bat.")
                self._note(f"ERROR: {e}")
            finally:
                self.busy = False

    def _one(self, a):
        t = a["type"]
        if t == "stop":
            return self.stop()
        if t == "home":
            return self.arm.home()
        if t == "gripper":
            return self.arm.gripper(a["value"])
        if t == "wait":
            end = time.time() + a["seconds"]
            while time.time() < end and not self.stop_flag.is_set():
                time.sleep(0.05)
            return
        p = self.arm.pose()
        x, y, z, tx, ty, tz = p["x"], p["y"], p["z"], p["tx"], p["ty"], p["tz"]
        if t == "move":
            x, y, z = x + a["dx"], y + a["dy"], z + a["dz"]
        elif t == "move_to":
            x, y, z = a.get("x", x), a.get("y", y), a.get("z", z)
        elif t == "rotate":
            tx, ty, tz = tx + a["droll"], ty + a["dpitch"], tz + a["dyaw"]
        _check_inside((x, y, z), (p["x"], p["y"], p["z"]))
        self.arm.move_to(x, y, z, tx, ty, tz)
