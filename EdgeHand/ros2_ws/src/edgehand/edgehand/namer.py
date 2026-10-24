"""Gemma names the tools that vision.find_tools found: a tray image with each tool's number on its handle + what each
tool looks like in words (config.TOOL_DESCRIPTIONS). Gemma picks a name per number; geometry stays with find_tools,
so Gemma never outputs coordinates (its own pointing was off by a median 246 mm in sim). Names are whitelisted
against config.TOOLS.
"""
import tempfile
from pathlib import Path
import cv2
import numpy as np
from edgehand import config, gemma

SYSTEM = """You identify tools for a robot arm from an overhead photo of a tool tray (dark mat)."""

ASK_TRAY = """Each tool has a white circle with a black number on its handle.
Tools on this bench:
{tools}
For each numbered tool ({ids}), say which tool it is: one of {names}, or unknown.
Use unknown when a tool doesn't match any description, for example a handle colour that isn't listed.
Reply ONLY with JSON like {{"1": "{first}", "2": "unknown"}}"""


def describe():
    return "\n".join(f"- {n}: {config.TOOL_DESCRIPTIONS.get(n, n)}" for n in config.TOOLS)


def marked_image(frame, tools, margin=60):
    """Each tool's number on its handle (so it can't be read as belonging to a neighbour), cropped to the tools."""
    out = frame.copy()
    boxes = []
    for i, t in enumerate(tools, 1):
        p = tuple(int(v) for v in t["handle_px"])
        cv2.circle(out, p, 13, (255, 255, 255), -1)
        cv2.circle(out, p, 13, (0, 0, 0), 2)
        cv2.putText(out, str(i), (p[0] - (7 if i < 10 else 12), p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(t["contour"])
        tip = t["grasp_px"] + (t["grasp_px"] - t["handle_px"])
        boxes += [(x, y), (x + w, y + h), tuple(tip)]
    b = np.array(boxes)
    x0, y0 = np.maximum(b.min(axis=0) - margin, 0).astype(int)
    x1, y1 = np.minimum(b.max(axis=0) + margin, [frame.shape[1], frame.shape[0]]).astype(int)
    return out[y0:y1, x0:x1]


def parse_names(d, n):
    """Gemma's {"1": "phillips", ...} -> list of n names (None for unknown / not whitelisted / missing)."""
    out = []
    for i in range(1, n + 1):
        v = (d or {}).get(str(i))
        v = v.strip().lower() if isinstance(v, str) else None
        out.append(v if v in config.TOOLS else None)
    return out


class GemmaNamer:
    def __init__(self, model=config.VISION_GEMMA_MODEL):
        self.engine = gemma.make_engine(model, vision=True)
        self.dir = Path(tempfile.gettempdir()) / "edgehand_namer"
        self.dir.mkdir(exist_ok=True)

    def _prompt(self, template, n):
        names = list(config.TOOLS)
        return template.format(tools=describe(), ids=", ".join(map(str, range(1, n + 1))), names=", ".join(names),
                               first=names[0])

    def name(self, frame, tools):
        """Names aligned with tools (None = unknown), from one numbered tray image. The image is left in self.dir."""
        if not tools:
            return []
        img = marked_image(frame, tools)
        path = self.dir / "tray.jpg"
        s = 768 / max(img.shape[:2])                         # Gemma's vision input size
        cv2.imwrite(str(path), cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)
        return parse_names(gemma.ask_json(self.engine, SYSTEM, self._prompt(ASK_TRAY, len(tools)), [path]), len(tools))
