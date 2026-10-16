import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="O Cabelo do Daniel")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import base64
    import io
    import json
    from PIL import Image
    return Image, base64, io, json, mo, np


@app.cell(hide_code=True)
def _(Image, base64, io, json, np):
    _PAYLOAD_B64 = "__PAYLOAD_B64__"
    DATA = json.loads(base64.b64decode(_PAYLOAD_B64))
    R = DATA["results"]

    def decode_png(b64, mode):
        raw = base64.b64decode(b64)
        return np.array(Image.open(io.BytesIO(raw)).convert(mode))

    PHOTOS = {}
    for _name, _p in DATA["photos"].items():
        PHOTOS[_name] = {
            "rgb": decode_png(_p["rgb_png_b64"], "RGB"),
            "labels": decode_png(_p["labels_png_b64"], "L"),
        }
    return DATA, PHOTOS, R, decode_png


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
# \U0001F52C O Cabelo do Daniel

### A rigorous, peer-unreviewed quantification of hair loss

Daniel Pereira submitted two photographs for independent scalp
analysis. What follows is a computer-vision pipeline applied with
complete sincerity to a question of no scientific importance
whatsoever: **exactly how much less hair does Daniel have now?**

Everything below the fold is interactive — the sliders recompute
the segmentation live, in your browser, on the original pixels.
"""
    )
    return


@app.cell(hide_code=True)
def _(Image, PHOTOS, mo, np):
    def to_image(arr):
        return Image.fromarray(arr.astype("uint8"))

    _before = to_image(PHOTOS["before"]["rgb"])
    _after = to_image(PHOTOS["after"]["rgb"])

    mo.hstack(
        [
            mo.vstack([mo.md("**Before** — lab, Bruker 400, allegedly a Tuesday"), mo.image(_before, width=360)]),
            mo.vstack([mo.md("**After** — professional headshot, zero plausible deniability"), mo.image(_after, width=360)]),
        ],
        justify="center",
        gap=2,
    )
    return (to_image,)


@app.cell(hide_code=True)
def _(R, mo):
    _b = R["before"]
    _a = R["after"]
    _delta_pp = R["coverage_delta_pp"]
    _rel = R["coverage_relative_change_pct"]

    def stat_card(label, value, sub=""):
        return mo.md(
            "<div style='border:1px solid var(--gray-5, #ccc); border-radius:12px; padding:14px 18px; min-width:190px;'>"
            "<div style='font-size:0.8rem; opacity:0.65; text-transform:uppercase; letter-spacing:0.04em;'>"
            + label
            + "</div>"
            + "<div style='font-size:1.6rem; font-weight:700; margin-top:4px;'>"
            + value
            + "</div>"
            + "<div style='font-size:0.8rem; opacity:0.6; margin-top:2px;'>"
            + sub
            + "</div></div>"
        )

    mo.hstack(
        [
            stat_card("Hair coverage — before", f"{_b['coverage']*100:.1f}%", f"95% CI [{_b['coverage_ci95'][0]*100:.1f}, {_b['coverage_ci95'][1]*100:.1f}]"),
            stat_card("Hair coverage — after", f"{_a['coverage']*100:.1f}%", f"95% CI [{_a['coverage_ci95'][0]*100:.1f}, {_a['coverage_ci95'][1]*100:.1f}]"),
            stat_card("Relative change", f"{_rel:.0f}%", f"{_delta_pp:.1f} percentage points lost"),
            stat_card("Hairline", "Stable" if R["hairline_stable"] else "Receded", "pure shearing event, not androgenetic"),
        ],
        justify="center",
        gap=1,
        wrap=True,
    )
    return (stat_card,)


@app.cell(hide_code=True)
def _(R, mo):
    _b, _a = R["before"]["coverage"] * 100, R["after"]["coverage"] * 100
    _lo_b, _hi_b = [v * 100 for v in R["before"]["coverage_ci95"]]
    _lo_a, _hi_a = [v * 100 for v in R["after"]["coverage_ci95"]]
    _max = max(_hi_b, _hi_a) * 1.25

    def bar(x, y, w, h, color):
        return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{color}' rx='6'/>"

    _chart_h = 220
    _chart_w = 420
    _bar_w = 110

    def to_px(v):
        return _chart_h - (v / _max) * (_chart_h - 20)

    _svg_parts = [
        f"<svg viewBox='0 0 {_chart_w} {_chart_h+40}' width='100%' style='max-width:480px; font-family: ui-sans-serif, system-ui;'>",
        f"<line x1='40' y1='{_chart_h}' x2='{_chart_w-10}' y2='{_chart_h}' stroke='currentColor' stroke-opacity='0.3'/>",
        bar(80, to_px(_b), _bar_w, _chart_h - to_px(_b), "#3B82F6"),
        bar(80 + _bar_w + 60, to_px(_a), _bar_w, _chart_h - to_px(_a), "#EF4444"),
        f"<line x1='{80+_bar_w/2}' y1='{to_px(_hi_b)}' x2='{80+_bar_w/2}' y2='{to_px(_lo_b)}' stroke='#1e3a8a' stroke-width='2'/>",
        f"<line x1='{80+_bar_w+60+_bar_w/2}' y1='{to_px(_hi_a)}' x2='{80+_bar_w+60+_bar_w/2}' y2='{to_px(_lo_a)}' stroke='#7f1d1d' stroke-width='2'/>",
        f"<text x='{80+_bar_w/2}' y='{to_px(_b)-10}' text-anchor='middle' font-size='15' font-weight='700' fill='currentColor'>{_b:.1f}%</text>",
        f"<text x='{80+_bar_w+60+_bar_w/2}' y='{to_px(_a)-10}' text-anchor='middle' font-size='15' font-weight='700' fill='currentColor'>{_a:.1f}%</text>",
        f"<text x='{80+_bar_w/2}' y='{_chart_h+22}' text-anchor='middle' font-size='13' fill='currentColor' opacity='0.75'>Before</text>",
        f"<text x='{80+_bar_w+60+_bar_w/2}' y='{_chart_h+22}' text-anchor='middle' font-size='13' fill='currentColor' opacity='0.75'>After</text>",
        "</svg>",
    ]
    _svg = "".join(_svg_parts)
    mo.vstack([mo.md("### Hair coverage, with 95% bootstrap confidence intervals (n=5000 resamples)"), mo.Html(_svg)], align="center")
    return bar, to_px


@app.cell(hide_code=True)
def _(mo):
    mo.md("## \U0001F39B️ Try it yourself: live scalp reclassification")
    threshold_slider = mo.ui.slider(start=-30, stop=30, step=2, value=0, label="Skin-tone threshold adjustment (Cr/Cb, more negative = stricter 'skin' call)")
    threshold_slider
    return (threshold_slider,)


@app.cell(hide_code=True)
def _(PHOTOS, mo, np, threshold_slider, to_image):
    def reclassify(name, delta):
        rgb = PHOTOS[name]["rgb"].astype(np.float64)
        labels = PHOTOS[name]["labels"]
        foreground = labels > 0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + 128
        cb = (b - y) * 0.564 + 128
        skin = (
            (cr >= 133 - delta) & (cr <= 178 + delta)
            & (cb >= 77 - delta) & (cb <= 128 + delta)
            & foreground
        )
        hair = foreground & ~skin
        coverage = hair.sum() / max(foreground.sum(), 1)

        overlay = PHOTOS[name]["rgb"].copy()
        overlay[hair] = (overlay[hair] * 0.35 + np.array([239, 68, 68]) * 0.65).astype("uint8")
        overlay[skin & foreground] = (overlay[skin & foreground] * 0.7 + np.array([34, 197, 94]) * 0.3).astype("uint8")
        return coverage, to_image(overlay)

    _delta = threshold_slider.value
    _cov_before, _img_before = reclassify("before", _delta)
    _cov_after, _img_after = reclassify("after", _delta)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([mo.md(f"**Before** — {_cov_before*100:.1f}% hair"), mo.image(_img_before, width=320)]),
                    mo.vstack([mo.md(f"**After** — {_cov_after*100:.1f}% hair"), mo.image(_img_after, width=320)]),
                ],
                justify="center",
                gap=2,
            ),
            mo.md(
                f"*Reclassified live from the original pixels at threshold offset **{_delta:+d}**. "
                "The head-silhouette mask (what counts as scalp vs. background) is fixed from the original "
                "GrabCut segmentation — only the hair/skin call within it is being redone in your browser.*"
            ),
        ]
    )
    return (reclassify,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## ⏳ Extrapolation (satirical, n=1, please do not cite)")
    years_slider = mo.ui.slider(start=1, stop=15, step=1, value=4, label="Assumed years between photos")
    years_slider
    return (years_slider,)


@app.cell(hide_code=True)
def _(R, mo, np, years_slider):
    _b = R["before"]["coverage"]
    _a = R["after"]["coverage"]
    _years = years_slider.value
    _rate = np.log(max(_a, 1e-4) / _b) / _years
    if _rate >= 0:
        _verdict = "Model refuses to extrapolate: coverage went up. Congratulations, recheck your photo order."
    else:
        _years_to_zero = np.log(0.001 / _b) / _rate
        _verdict = (
            f"At this rate, Daniel's modeled scalp hair coverage crosses below 0.1% "
            f"approximately **{_years_to_zero:.1f} years** after the 'before' photo — "
            f"i.e. around **{_years_to_zero - _years:.1f} years from the 'after' photo**. "
            "Model: exponential decay, fit to n=2 points, zero statistical power, maximum confidence."
        )
    mo.callout(mo.md(_verdict), kind="warn")
    return


@app.cell(hide_code=True)
def _(R, mo):
    mo.md(
        f"""
---
## Methods (abridged, for the curious)

1. **Face + eye detection** — Haar cascades (OpenCV) locate each face
   and both eyes.
2. **Scale calibration** — pixel distances are converted to millimetres
   using the adult average interpupillary distance ({R['ipd_reference_mm']:.0f}mm,
   Dodgson 2004) measured between the detected pupils.
3. **Head segmentation** — GrabCut, seeded from an elliptical prior
   fit to the face box, separates scalp from background per photo
   (background varies: studio white vs. lab wall).
4. **Hair vs. visible-skin classification** — YCrCb skin-tone
   thresholding within the segmented head region.
5. **Uncertainty** — 95% confidence intervals via {R['n_bootstrap']}-resample
   pixel bootstrap.
6. **Hairline** — row-wise scan for the first mostly-skin row.
   {R['hairline_note']}

Full source, including this notebook's generator, is reproducible via
`uv run python -m ocabelododaniel.pipeline` in the repository.

*Not peer reviewed. Daniel did not consent to the GrabCut parameters.*
"""
    )
    return


if __name__ == "__main__":
    app.run()
