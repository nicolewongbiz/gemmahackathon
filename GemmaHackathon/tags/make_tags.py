"""Regenerate tags/apriltags.pdf: tag36h11 IDs from config, black square = TAG_SIZE_M, US Letter.
    python tags/make_tags.py
"""
import sys
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

DPI = 300
PAGE_W, PAGE_H = int(8.5 * DPI), int(11 * DPI)
MM = DPI / 25.4
TAG_PX = round(config.TAG_SIZE_M * 1000 * MM)       # black square edge
QUIET_PX = round(TAG_PX / 8)                         # 1-cell white margin (detector needs it), cut line outside it
ROW_PITCH = int(3.08 * DPI)

PLACEMENT = {0: "table corner near arm, RIGHT", 1: "table corner near arm, LEFT",
             2: "table corner far, LEFT", 3: "table corner far, RIGHT"}
TAGS = {i: PLACEMENT[i] for i in config.TABLE_TAGS}
TAGS.update({i: f"on BIN {name}" for i, name in config.BIN_TAGS.items()})


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def page(title):
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    d = ImageDraw.Draw(img)
    d.text((DPI // 2, DPI // 3), title, fill=0, font=font(48))
    d.text((DPI // 2, DPI // 3 + 70),
           f"Print at 100% / Actual size (NOT fit to page). Black square must measure {config.TAG_SIZE_M * 100:.1f} cm.",
           fill=0, font=font(34))
    # 5 cm check bar
    y, x0 = DPI // 3 + 140, DPI // 2
    d.rectangle([x0, y, x0 + TAG_PX, y + 20], fill=0)
    d.text((x0 + TAG_PX + 30, y - 10), f"<- this bar = {config.TAG_SIZE_M * 100:.1f} cm", fill=0, font=font(34))
    return img, d


def draw_tag(img, d, tid, label, cx, top):
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tid, 8)
    tag = Image.fromarray(marker).resize((TAG_PX, TAG_PX), Image.NEAREST)
    x = cx - TAG_PX // 2
    img.paste(tag, (x, top))
    o = QUIET_PX
    d.rectangle([x - o, top - o, x + TAG_PX + o, top + TAG_PX + o], outline=160, width=2)  # cut line
    d.text((x - o, top + TAG_PX + o + 15), f"ID {tid}", fill=0, font=font(56))
    d.text((x - o, top + TAG_PX + o + 85), label, fill=0, font=font(34))


def build():
    pages = []
    ids = list(TAGS)
    for title in ("AprilTags tag36h11 - Show Me Once", "AprilTags tag36h11 - SPARES"):
        img, d = page(title)
        for k, tid in enumerate(ids):
            col, row = k % 2, k // 2
            draw_tag(img, d, tid, TAGS[tid], cx=PAGE_W // 4 + col * PAGE_W // 2, top=int(1.35 * DPI) + row * ROW_PITCH)
        pages.append(img)
    out = Path(__file__).parent / "apriltags.pdf"
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=DPI)
    return out, pages


if __name__ == "__main__":
    out, _ = build()
    print("wrote", out)
