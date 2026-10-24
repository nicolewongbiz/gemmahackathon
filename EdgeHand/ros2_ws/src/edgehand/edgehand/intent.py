"""Operator text -> {"tool": <class in TOOLS> | None, "reply": str}, using Gemma 4 on LiteRT-LM.

Gemma only names a tool class; it never produces coordinates. Output is whitelisted against config.TOOLS.
"""
import json, re
from edgehand import config

SYSTEM = """You are the voice of a robot arm that hands tools to a technician.
Tools you can hand over: {tools}.
Map the request to ONE tool from that list, or null if they didn't ask for one of them.
Synonyms: "cross-head", "plus" = phillips; "slotted", "flat", "straight" = flathead; "star" = torx.
Reply ONLY with JSON: {{"tool": "<tool or null>", "reply": "<very short spoken reply>"}}"""


def parse_reply(text):
    """First {...} in Gemma's text -> whitelisted dict. Raises ValueError if there's no JSON."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in: {text[:200]}")
    d = json.loads(m.group(0))
    tool = d.get("tool")
    tool = tool.strip().lower() if isinstance(tool, str) else None
    if tool not in config.TOOLS:
        tool = None
    return {"tool": tool, "reply": str(d.get("reply", "")).strip()}


class GemmaIntent:
    def __init__(self, model=config.GEMMA_MODEL):
        import litert_lm
        B = litert_lm.Backend
        try:
            self.engine = litert_lm.Engine(model, backend=B.GPU if config.GEMMA_USE_GPU else B.CPU)
        except Exception:
            self.engine = litert_lm.Engine(model, backend=B.CPU)
        self.system = SYSTEM.format(tools=", ".join(config.TOOLS))

    def _ask(self, text):
        with self.engine.create_conversation(messages=[{"role": "system", "content": self.system}]) as conv:
            r = conv.send_message({"role": "user", "content": [{"type": "text", "text": text}]})
        return r["content"][0]["text"]

    def parse(self, text):
        prompt, last = text, ""
        for _ in range(config.GEMMA_RETRIES + 1):
            last = self._ask(prompt)
            try:
                return parse_reply(last)
            except (ValueError, json.JSONDecodeError):
                prompt = text + "\nReply with ONLY a valid JSON object."
        return {"tool": None, "reply": "Sorry, I didn't get that."}


if __name__ == "__main__":
    import sys, time
    g = GemmaIntent(sys.argv[1] if len(sys.argv) > 1 else config.GEMMA_MODEL)
    for line in sys.stdin:
        t0 = time.time()
        print(g.parse(line.strip()), f"({time.time() - t0:.1f} s)", flush=True)
