"""One-command reproducible build: analysis -> notebook -> static site.

    uv run python -m ocabelododaniel.build

Runs the CV pipeline (output/results.json, output/masks.npz), bakes the
results into a self-contained marimo notebook (webapp/app.py), and exports
that notebook to a static WASM-powered site (docs/) ready for GitHub Pages.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import make_app, pipeline

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def main() -> None:
    print("== 1/3 running analysis pipeline ==")
    pipeline.main()

    print("== 2/3 generating marimo notebook ==")
    make_app.main()

    print("== 3/3 exporting static WASM site ==")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(ROOT / "webapp" / "app.py"),
            "-o",
            str(DOCS),
            "--mode",
            "run",
            "-f",
        ],
        check=True,
        cwd=ROOT,
    )

    # marimo ships a CLAUDE.md editing-helper alongside the export; it's not
    # part of the published site.
    stray = DOCS / "CLAUDE.md"
    if stray.exists():
        stray.unlink()

    print(f"done. Open {DOCS / 'index.html'} (serve over HTTP, e.g. `python -m http.server --directory docs`).")


if __name__ == "__main__":
    main()
