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
2. Splits "how much less hair?" into two measurable questions:
   - **Did the hairline move?** Each photo is levelled and scaled on the
     eyes, then the hairline is traced per forehead column as the steepest
     drop in colour saturation. Heights are measured relative to a small
     forehead **scar** visible in both photos, which cancels the different
     head pitch / camera angle (the *after* photo is 16% foreshortened).
   - **How much shorter is the hair?** The hair zone just behind the
     hairline is compared with forehead skin just in front of it in linear
     RGB (lighting cancels), unmixed jointly across both photos into hair
     opacity, and turned into a length ratio with a Poisson shaft-coverage
     model in which the unknown density/geometry constant cancels.
   - 2000-resample bootstrap over forehead columns for 95% CIs.
3. Bakes the results into a single-file, dependency-free **marimo** notebook
   that runs entirely client-side (Pyodide/WASM): a scar-aligned
   before/after cross-fade, the saturation profiles on a common anatomical
   axis, live sensitivity sliders for the model's one unmeasurable
   assumption, a clipper-guard → grams-of-hair converter, and a satirical
   extrapolation.
4. Exports that notebook as a static site in `docs/`, served by GitHub Pages.

Headline findings: **hairline shift +1.2 mm (95% CI −1.7 to +3.1 mm)** — no
recession; **hair ~49% shorter (95% CI 45–76%)**. Daniel is not going bald.
Daniel is going short.

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
  pipeline.py            CV analysis -> output/results.json, output/assets.npz
  make_app.py             bakes output/ into webapp/app.py (self-contained)
  templates/app_template.py  the marimo notebook source (payload gets inlined)
  build.py                 runs all of the above + the WASM export
output/, webapp/, docs/    generated (gitignored except docs/, which GitHub
                            Pages serves)
```

## Methodology caveats (read: this is a joke, rigorously executed)

- n = 1. Bootstrap intervals quantify measurement noise across the forehead,
  not anything about anyone else's scalp.
- Only the length *ratio* is measured; absolute millimetres need one known
  clipper guard (the site lets you pick one).
- The *before* buzz is close to opaque, which makes the length interval
  lopsided. The model's one unmeasured assumption (is the scalp paler than
  the forehead?) is exposed as a slider so you can see how much it matters.
- The extrapolation section is a two-point exponential fit. It is included
  because it is funny, not because it is true.
- Erratum: v1 classified stubble as bare skin, reported an 88% "hair loss"
  and hinted at a receding hairline. Both were wrong; see the site.
