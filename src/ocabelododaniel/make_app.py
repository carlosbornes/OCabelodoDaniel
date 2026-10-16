"""Bakes output/results.json + output/assets.npz into a single, dependency-free
marimo notebook (webapp/app.py) with all data embedded as base64, so it can be
exported with `marimo export html-wasm` and run entirely client-side.
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
DISPLAY_WIDTH = 520

IMAGES = ["before", "after_matched", "before_annot", "after_annot", "after_matched_annot"]
ARRAYS = ["profile_u", "profile_before", "profile_after", "columns_before", "columns_after"]


def jpeg_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    h = round(img.height * DISPLAY_WIDTH / img.width)
    img = img.resize((DISPLAY_WIDTH, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    results = json.loads((OUT / "results.json").read_text())
    npz = np.load(OUT / "assets.npz")
    payload = {
        "results": results,
        "images": {k: jpeg_b64(npz[k]) for k in IMAGES},
        "arrays": {k: npz[k].tolist() for k in ARRAYS},
    }
    payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    content = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__PAYLOAD_B64__", payload_b64)

    WEBAPP.mkdir(exist_ok=True)
    app_path = WEBAPP / "app.py"
    app_path.write_text(content, encoding="utf-8")
    print(f"wrote {app_path} ({app_path.stat().st_size / 1_048_576:.2f} MB)")


if __name__ == "__main__":
    main()
