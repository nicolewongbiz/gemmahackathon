"""Teach / plan / correct logic on top of gemma_vision."""
import json
import config
from planner import gemma_vision, prompts

RULE_FILE = config.DATA / "rule.json"


class Brain:
    def __init__(self):
        self.rule = {"rule": "unknown", "bins": {"A": "?", "B": "?"}}
        if RULE_FILE.exists():
            self.rule = json.loads(RULE_FILE.read_text())

    def _save(self):
        RULE_FILE.write_text(json.dumps(self.rule, indent=2))

    def teach(self, keyframes):
        out = gemma_vision.ask_json(prompts.TEACH.format(n=len(keyframes)), keyframes, prompts.SYSTEM)
        self.rule = {"rule": out.get("rule", ""), "bins": out.get("bins", {})}
        self._save()
        return out.get("say") or f"Got it. {self.rule['rule']}"

    def plan(self, marked_frame, ids):
        p = prompts.PLAN.format(rule=self.rule["rule"], bin_a=self.rule["bins"].get("A"),
                                bin_b=self.rule["bins"].get("B"), ids=ids)
        out = gemma_vision.ask_json(p, [marked_frame], prompts.SYSTEM)
        if out.get("object") not in ids:
            out["object"] = None
        return out

    def correct(self, text):
        p = prompts.CORRECT.format(rule=self.rule["rule"], bin_a=self.rule["bins"].get("A"),
                                   bin_b=self.rule["bins"].get("B"), text=text)
        out = gemma_vision.ask_json(p, [], prompts.SYSTEM)
        self.rule = {"rule": out.get("rule", self.rule["rule"]), "bins": out.get("bins", self.rule["bins"])}
        self._save()
        return out.get("say", "Updated.")
