"""Test Gemma without the robot.

    python scripts/check_gemma.py                    # uses config.BACKEND
    python scripts/check_gemma.py --backend ollama
    python scripts/check_gemma.py --probe            # print what the installed litert_lm offers
"""
import argparse, inspect, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["litert", "ollama", "rules"])
ap.add_argument("--probe", action="store_true")
a = ap.parse_args()

if a.probe:
    import litert_lm
    print("litert_lm version:", getattr(litert_lm, "__version__", "?"))
    print("module members:", [n for n in dir(litert_lm) if not n.startswith("_")])
    try:
        print("Engine signature:", inspect.signature(litert_lm.Engine))
    except (TypeError, ValueError):
        print("Engine signature: (not introspectable)")
    if hasattr(litert_lm, "Backend"):
        print("Backend members:", [n for n in dir(litert_lm.Backend) if not n.startswith("_")])
    print("\nIf GPU fails, edit LiteRTBackend._make_engine() in gemma.py to match the names above.")
    sys.exit()

from gemma import make_backend, parse_reply  # noqa: E402

b = make_backend(a.backend)
for q in ["hi, who are you?", "move the hand 5 cm forward",
          "go up 10 cm, then left 3 cm, then close the gripper", "rotate the wrist 30 degrees and go home"]:
    t = time.time()
    raw = b.send(q + "\n\n[robot state] hand at x=0.450 y=0.000 z=0.400 m, gripper=0.0, busy=False")
    print(f"\n> {q}\n  ({time.time()-t:.2f}s) raw: {raw[:300]}")
    print("  parsed:", json.dumps(parse_reply(raw)))
