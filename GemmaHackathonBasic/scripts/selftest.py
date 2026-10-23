"""Offline self-test: no robot, no model. python scripts/selftest.py"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
config.MOCK_ARM = True
from arm import MockArm  # noqa: E402
from commands import Executor, SafetyError, validate, describe  # noqa: E402
from gemma import RulesBackend, parse_reply  # noqa: E402

# parse_reply
assert parse_reply('```json\n{"reply":"hi","actions":[]}\n```')["reply"] == "hi"
assert parse_reply("Sure! {\"reply\": \"ok\", \"actions\": [{\"type\": \"home\"}]} done")["actions"] == [{"type": "home"}]
assert parse_reply("not json at all") == {"reply": "not json at all", "actions": []}

# rules backend
r = parse_reply(RulesBackend().send("move the hand 5 cm forward and 20 mm up"))
assert abs(r["actions"][0]["dx"] - 0.05) < 1e-9 and abs(r["actions"][1]["dz"] - 0.02) < 1e-9, r

# validation
try:
    validate([{"type": "move", "dx": 0.5}]); raise AssertionError("should block 50 cm")
except SafetyError:
    pass
assert validate([{"type": "gripper", "value": "close"}])[0]["value"] == 1.0
print(describe(validate([{"type": "move", "dx": 0.05, "dy": -0.02}])[0]))

# executor + workspace
ex = Executor(MockArm())
ex.submit([{"type": "move", "dx": 0.05}, {"type": "gripper", "value": 1.0}])
while ex.busy or not ex.log or ex.log[-1].split()[-1] != "done":
    time.sleep(0.05)
assert abs(ex.arm.pose()["x"] - 0.50) < 1e-9
ex.submit([{"type": "move", "dz": -0.19}, {"type": "move", "dz": -0.19}, {"type": "move", "dz": -0.19}])
time.sleep(2.5)
assert any("outside the workspace" in l for l in ex.log), ex.log[-3:]
print("\n".join(ex.log))
print("\nSELFTEST PASSED")
