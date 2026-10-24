"""Operator text -> {"tool": <class in TOOLS> | None, "reply": str}, using Gemma 4 on LiteRT-LM.

Gemma only names a tool class; it never produces coordinates. Output is whitelisted against config.TOOLS.
"""
from edgehand import config, gemma

SYSTEM = """You are the voice of a robot arm that hands tools to a technician.
Tools you can hand over: {tools}.
Map the request to ONE tool from that list, or null if they didn't ask for one of them.
Synonyms: "cross-head", "plus" = phillips; "slotted", "flat", "straight" = flathead; "star" = torx.
If they describe a job instead of naming a tool ("I need to pry this open"), pick the tool from the list that does
that job. Only use null when no tool on the list fits.
Reply ONLY with JSON: {{"tool": "<tool or null>", "reply": "<very short spoken reply>"}}"""


def parse_reply(text):
    """First {...} in Gemma's text -> whitelisted dict. Raises ValueError if there's no JSON."""
    return whitelist(gemma.first_json(text))


def whitelist(d):
    tool = d.get("tool")
    tool = tool.strip().lower() if isinstance(tool, str) else None
    if tool not in config.TOOLS:
        tool = None
    return {"tool": tool, "reply": str(d.get("reply", "")).strip()}


class GemmaIntent:
    def __init__(self, model=config.GEMMA_MODEL):
        self.engine = gemma.make_engine(model)
        self.system = SYSTEM.format(tools=", ".join(config.TOOLS))

    def parse(self, text):
        d = gemma.ask_json(self.engine, self.system, text)
        return whitelist(d) if d is not None else {"tool": None, "reply": "Sorry, I didn't get that."}


if __name__ == "__main__":
    import sys, time
    g = GemmaIntent(sys.argv[1] if len(sys.argv) > 1 else config.GEMMA_MODEL)
    for line in sys.stdin:
        t0 = time.time()
        print(g.parse(line.strip()), f"({time.time() - t0:.1f} s)", flush=True)
