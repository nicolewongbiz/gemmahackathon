"""Teach / plan / correct: prompts + rule state on top of gemma_vision."""
import json
import config
from planner import gemma_vision

RULE_FILE = config.DATA / "rule.json"

SYSTEM = (
    "You are the planner for a tabletop robot arm. Images show a table from above. "
    "Objects are marked with numbered white circles. Bins are labelled A and B. "
    "Always answer with a single JSON object and nothing else."
)

TEACH = """These {n} frames were taken in order while a person sorted objects into bins.
Work out the sorting RULE they used (e.g. by category, colour, material, size).
Return JSON: {{"rule": "<one sentence>", "bins": {{"A": "<what goes in A>", "B": "<what goes in B>"}}, "say": "<short sentence to speak aloud>"}}"""

PLAN = """Sorting rule: {rule}
Bin A: {bin_a}. Bin B: {bin_b}.
Numbered objects still on the table: {ids}.
Pick ONE object to move next and choose its bin.
Return JSON: {{"object": <number>, "label": "<what it is>", "bin": "A" or "B", "reason": "<few words>"}}
If no numbered object fits either bin, return {{"object": null, "say": "<why>"}}"""

CORRECT = """Current rule: {rule}. Bin A: {bin_a}. Bin B: {bin_b}.
The user said: "{text}"
Update the rule. Return JSON: {{"rule": "...", "bins": {{"A": "...", "B": "..."}}, "say": "<confirm in one sentence>"}}"""


class Brain:
    def __init__(self):
        self.rule = {"rule": "unknown", "bins": {"A": "?", "B": "?"}}
        if RULE_FILE.exists():
            self.rule = json.loads(RULE_FILE.read_text())

    def _save(self):
        RULE_FILE.write_text(json.dumps(self.rule, indent=2))

    def teach(self, keyframes):
        out = gemma_vision.ask_json(TEACH.format(n=len(keyframes)), keyframes, SYSTEM)
        self.rule = {"rule": out.get("rule", ""), "bins": out.get("bins", {})}
        self._save()
        return out.get("say") or f"Got it. {self.rule['rule']}"

    def plan(self, marked_frame, ids):
        p = PLAN.format(rule=self.rule["rule"], bin_a=self.rule["bins"].get("A"),
                        bin_b=self.rule["bins"].get("B"), ids=ids)
        out = gemma_vision.ask_json(p, [marked_frame], SYSTEM)
        if out.get("object") not in ids:
            out["object"] = None
        return out

    def correct(self, text):
        p = CORRECT.format(rule=self.rule["rule"], bin_a=self.rule["bins"].get("A"),
                           bin_b=self.rule["bins"].get("B"), text=text)
        out = gemma_vision.ask_json(p, [], SYSTEM)
        self.rule = {"rule": out.get("rule", self.rule["rule"]), "bins": out.get("bins", self.rule["bins"])}
        self._save()
        return out.get("say", "Updated.")
