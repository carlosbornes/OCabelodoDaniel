# O Cabelo do Daniel 🔬

A fully reproducible, scientifically-toned, legally-meaningless quantification
of how much less hair Daniel Pereira has now than he
used to. Built because he deserved better than a vague "bro you're going
bald."

**Live site:** _published via GitHub Pages — see Settings → Pages once this
repo is pushed._

## What this actually does

1. Takes one composite photo (`data/hair.png`, a casual lab photo next to a
   much-more-shaved formal headshot).
2. Runs a real (if gleefully overkill) computer-vision pipeline on it:
   Haar-cascade face/eye detection, millimetre scale calibration from the
   adult average interpupillary distance, GrabCut head-silhouette
   segmentation, YCrCb hair/skin classification, and a 5000-resample pixel
   bootstrap for 95% confidence intervals.
3. Bakes the results into a single-file, dependency-free **marimo** notebook
   that runs entirely client-side (Pyodide/WASM) — no server, fully
   interactive: live threshold sliders recompute the hair/skin call on the
   original pixels in your browser, plus a satirical exponential-decay
   "years to total baldness" extrapolator.
4. Exports that notebook as a static site in `docs/`, served by GitHub Pages.

Headline finding: **~88% relative reduction in visible scalp hair coverage**
(18.8% → 2.2% of the segmented head region), hairline position statistically
unchanged (it's a buzz, not a recession).

## Reproducing it

Everything is pinned and run through [`uv`](https://docs.astral.sh/uv/):

```bash
uv run python -m ocabelododaniel.build
```

This single command regenerates `output/` (analysis results + masks),
`webapp/app.py` (the generated marimo notebook), and `docs/` (the exported
static site) from `data/hair.png`, deterministically (fixed RNG seed for the
bootstrap).

To preview the site locally:

```bash
python -m http.server --directory docs
```

To iterate on just the notebook interactively (regular marimo, not WASM):

```bash
uv run marimo edit webapp/app.py
```

### Project layout

```
data/hair.png           the one input
src/ocabelododaniel/
  pipeline.py            CV analysis -> output/results.json, output/masks.npz
  make_app.py             bakes output/ into webapp/app.py (self-contained)
  templates/app_template.py  the marimo notebook source (payload gets inlined)
  build.py                 runs all of the above + the WASM export
output/, webapp/, docs/    generated (gitignored except docs/, which GitHub
                            Pages serves)
```

## Methodology caveats (read: this is a joke, rigorously executed)

- n = 1. Bootstrap confidence intervals quantify pixel-classification noise,
  not whether this generalizes to anyone else's scalp, or indeed to Daniel's
  scalp at any other time.
- The two photos differ in lighting, camera, and background; the pipeline
  adapts per-photo (GrabCut re-segments each one independently) but is not
  immune to all of that.
- The "years to total baldness" model is a two-point exponential fit. Two
  points determine a line (or a curve) perfectly and mean nothing
  statistically. It is included because it is funny, not because it is true.
- Daniel did not review or approve the GrabCut parameters used against his
  forehead.
