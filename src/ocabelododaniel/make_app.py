"""Bakes output/results.json + output/masks.npz into a single, dependency-free
marimo notebook (webapp/app.py) with all image/array data embedded as base64.

The notebook has zero local-file or local-module dependencies -- everything it
needs lives in one literal string inside the file -- so it can be exported
with `marimo export html-wasm` and run entirely client-side via Pyodide,
with no server and no access to this repo's data/ or output/ directories.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
WEBAPP = ROOT / "webapp"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "app_template.py"
DISPLAY_WIDTH = 480  # downscale crops for a fast, lightweight page


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def resize(arr: np.ndarray, mode: str) -> Image.Image:
    img = Image.fromarray(arr, mode=mode)
    if img.width > DISPLAY_WIDTH:
        new_h = round(img.height * DISPLAY_WIDTH / img.width)
        img = img.resize((DISPLAY_WIDTH, new_h), Image.LANCZOS if mode == "RGB" else Image.NEAREST)
    return img


def build_payload() -> dict:
    results = json.loads((OUT / "results.json").read_text())
    npz = np.load(OUT / "masks.npz")

    photos = {}
    for name in ("before", "after"):
        rgb_img = resize(npz[f"{name}_rgb"], "RGB")
        labels = np.zeros(npz[f"{name}_hair"].shape, dtype=np.uint8)
        labels[npz[f"{name}_skin"]] = 1
        labels[npz[f"{name}_hair"]] = 2
        labels_img = resize(labels, "L")
        photos[name] = {
            "rgb_png_b64": png_b64(rgb_img),
            "labels_png_b64": png_b64(labels_img),
            "width": rgb_img.width,
            "height": rgb_img.height,
        }

    return {"results": results, "photos": photos}


def main() -> None:
    payload = build_payload()
    payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    content = template.replace("__PAYLOAD_B64__", payload_b64)

    WEBAPP.mkdir(exist_ok=True)
    app_path = WEBAPP / "app.py"
    app_path.write_text(content, encoding="utf-8")
    print(f"wrote {app_path} ({app_path.stat().st_size / 1_048_576:.2f} MB)")


if __name__ == "__main__":
    main()
