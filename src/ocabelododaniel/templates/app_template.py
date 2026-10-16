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
    _data = json.loads(base64.b64decode(_PAYLOAD_B64))
    R = _data["results"]
    IMG = {k: Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB") for k, v in _data["images"].items()}
    ARR = {k: np.array(v) for k, v in _data["arrays"].items()}
    return ARR, IMG, R


@app.cell(hide_code=True)
def _(R, mo):
    _shift = R["hairline_shift_mm"]
    _short = R["hair_shorter_pct"]
    mo.md(
        f"""
# \U0001F52C O Cabelo do Daniel

### A rigorous, peer-unreviewed quantification of a buzzcut

Daniel Pereira supplied two photographs. We asked the only question that
matters, and split it in two, because rigour:

1. **Did the hairline move?** (i.e. is this follicle loss?)
2. **How much shorter is the hair?** (i.e. is this a haircut with commitment issues?)

**Findings.** The hairline moved **{_shift['estimate']:+.1f} mm**
(95% CI {_shift['ci95'][0]:+.1f} to {_shift['ci95'][1]:+.1f} mm) —
statistically indistinguishable from zero. The hair itself is
**{_short['estimate']:.0f}% shorter** (95% CI {_short['ci95'][0]:.0f}–{_short['ci95'][1]:.0f}%).
Daniel is not going bald. Daniel is going *short*.
"""
    )
    return


@app.cell(hide_code=True)
def _(R, mo):
    def _card(label, value, sub):
        return mo.Html(
            "<div style='border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:14px 18px;min-width:200px'>"
            f"<div style='font-size:.75rem;opacity:.65;text-transform:uppercase;letter-spacing:.05em'>{label}</div>"
            f"<div style='font-size:1.55rem;font-weight:700;margin-top:4px'>{value}</div>"
            f"<div style='font-size:.8rem;opacity:.65;margin-top:2px'>{sub}</div></div>"
        )

    _s, _l = R["hairline_shift_mm"], R["hair_shorter_pct"]
    _ob, _oa = R["opacity_before"], R["opacity_after"]
    mo.hstack(
        [
            _card("Hairline shift", f"{_s['estimate']:+.1f} mm", f"95% CI [{_s['ci95'][0]:+.1f}, {_s['ci95'][1]:+.1f}] — no recession"),
            _card("Hair length", f"−{_l['estimate']:.0f}%", f"95% CI [{_l['ci95'][0]:.0f}, {_l['ci95'][1]:.0f}]%"),
            _card("Scalp visible behind hairline", f"{(1-_ob['estimate'])*100:.0f}% → {(1-_oa['estimate'])*100:.0f}%", "before → after"),
            _card("Pose correction applied", f"{(1-R['foreshortening_after_vs_before'])*100:.0f}%", "vertical foreshortening, via scar fiducial"),
        ],
        justify="center",
        gap=1,
        wrap=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
## 1. The hairline did not move

Measuring a hairline in "pixels above the eyes" is a trap: the two photos were
taken at different head pitch and camera height, which squashes everything
above the eyes by a different amount in each. In the *after* photo, the
forehead looks 16% shorter — not because it is, but because of the angle.

So we need a ruler printed on Daniel's forehead. Fortunately, Daniel came
pre-equipped: **a small scar** (green box), present in both photos, sitting on
the same patch of forehead as the hairline. Heights measured *relative to the
scar* are immune to the pitch difference. The hairline (yellow) is traced
column by column as the steepest drop in colour saturation — skin is
saturated, hair is not.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    view = mo.ui.radio(
        options=["As photographed", "Anatomically aligned (scar-matched)"],
        value="Anatomically aligned (scar-matched)",
        inline=True,
    )
    view
    return (view,)


@app.cell(hide_code=True)
def _(IMG, mo, view):
    _after = IMG["after_annot"] if view.value == "As photographed" else IMG["after_matched_annot"]
    mo.hstack(
        [
            mo.vstack([mo.md("**Before** — lab, beside a Bruker magnet"), mo.image(IMG["before_annot"], width=340)]),
            mo.vstack([mo.md("**After** — professional headshot"), mo.image(_after, width=340)]),
        ],
        justify="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    fade = mo.ui.slider(start=0, stop=1, step=0.02, value=0.5, label="Before ← cross-fade → After (scar-aligned)", full_width=True)
    show_trace = mo.ui.checkbox(value=True, label="show hairline + scar tracing")
    mo.vstack([fade, show_trace])
    return fade, show_trace


@app.cell(hide_code=True)
def _(IMG, Image, fade, mo, show_trace):
    _a = IMG["before_annot"] if show_trace.value else IMG["before"]
    _b = IMG["after_matched_annot"] if show_trace.value else IMG["after_matched"]
    mo.hstack(
        [
            mo.image(Image.blend(_a, _b, fade.value), width=480),
            mo.md(
                "Slide back and forth. The eyes and scar stay put (that's the alignment), "
                "the **hairline stays put** (that's the finding), and the only thing that "
                "changes is how dark the hair zone is (that's the haircut)."
            ),
        ],
        align="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(ARR, R, mo):
    def _chart():
        u, pb, pa = ARR["profile_u"], ARR["profile_before"], ARR["profile_after"]
        W, H, L, T, B, Rm = 560, 300, 50, 16, 40, 16
        xmin, xmax = float(u.min()), float(u.max())
        ymax = float(max(pb.max(), pa.max())) * 1.1

        def sx(v):
            return L + (v - xmin) / (xmax - xmin) * (W - L - Rm)

        def sy(v):
            return T + (1 - v / ymax) * (H - T - B)

        def path(ys):
            return " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(zip(u, ys)))

        parts = [f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:600px;font-family:ui-sans-serif,system-ui'>"]
        for v in range(0, int(ymax) + 1, 10):
            parts.append(f"<line x1='{L}' x2='{W-Rm}' y1='{sy(v):.1f}' y2='{sy(v):.1f}' stroke='currentColor' stroke-opacity='.12'/>")
            parts.append(f"<text x='{L-6}' y='{sy(v)+4:.1f}' text-anchor='end' font-size='11' fill='currentColor' opacity='.7'>{v}</text>")
        for v in (0.75, 1.0, 1.25, 1.5):
            parts.append(f"<text x='{sx(v):.1f}' y='{H-B+16}' text-anchor='middle' font-size='11' fill='currentColor' opacity='.7'>{v:.2f}</text>")
        parts.append(f"<line x1='{sx(1.0):.1f}' x2='{sx(1.0):.1f}' y1='{T}' y2='{H-B}' stroke='#16a34a' stroke-dasharray='3 3'/>")
        parts.append(f"<text x='{sx(1.0)+4:.1f}' y='{T+12}' font-size='11' fill='#16a34a'>scar</text>")
        for key, col in (("hairline_before", "#2563eb"), ("hairline_after", "#ea580c")):
            hx = sx(R[key]["estimate"])
            lo, hi = sx(R[key]["ci95"][0]), sx(R[key]["ci95"][1])
            parts.append(f"<rect x='{lo:.1f}' width='{hi-lo:.1f}' y='{T}' height='{H-T-B}' fill='{col}' opacity='.10'/>")
            parts.append(f"<line x1='{hx:.1f}' x2='{hx:.1f}' y1='{T}' y2='{H-B}' stroke='{col}' stroke-width='1.5'/>")
        parts.append(f"<path d='{path(pb)}' fill='none' stroke='#2563eb' stroke-width='2.5'/>")
        parts.append(f"<path d='{path(pa)}' fill='none' stroke='#ea580c' stroke-width='2.5'/>")
        parts.append(f"<text x='{sx(0.62):.1f}' y='{sy(pb[2])-8:.1f}' font-size='12' font-weight='600' fill='#2563eb'>before</text>")
        parts.append(f"<text x='{sx(1.55):.1f}' y='{sy(pa[-30])-8:.1f}' font-size='12' font-weight='600' fill='#ea580c'>after</text>")
        parts.append(f"<text x='{(L+W-Rm)/2:.1f}' y='{H-6}' text-anchor='middle' font-size='12' fill='currentColor' opacity='.8'>height above eye line, in scar units (scar = 1)</text>")
        parts.append(f"<text x='14' y='{(T+H-B)/2:.1f}' font-size='12' fill='currentColor' opacity='.8' transform='rotate(-90 14 {(T+H-B)/2:.1f})' text-anchor='middle'>skin colour saturation (chroma)</text>")
        parts.append("</svg>")
        return "".join(parts)

    mo.vstack(
        [
            mo.md("### Colour saturation up the forehead, both photos on the same anatomical axis"),
            mo.Html(_chart()),
            mo.md(
                "Both curves fall off a cliff at the same place (vertical lines, with shaded 95% CIs): "
                "**same hairline**. But *before* falls to a much lower floor (dark, opaque hair) "
                "while *after* plateaus higher (stubble with scalp showing through): **shorter hair**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
## 2. The hair got shorter

Just behind the hairline, the hair zone is compared with the forehead skin just
in front of it, **in the same photo**, so lighting cancels. In linear RGB, a
patch where hair covers a fraction *f* of the scalp reflects

> *q* = *s*(1 − *f*) + *f* · *h*   (per colour channel, relative to forehead skin)

where *h* is hair-vs-skin albedo (same hair in both photos, so fitted jointly)
and *s* is scalp-vs-forehead albedo. For short stubble, shafts land roughly at
random, so coverage follows Poisson statistics: *f* = 1 − e<sup>−kL</sup>.
The constant *k* (follicle density × shaft diameter × viewing geometry) is
the same person and the same view in both photos, so it cancels in the ratio:

> *L*<sub>after</sub> / *L*<sub>before</sub> = ln(1 − *f*<sub>after</sub>) / ln(1 − *f*<sub>before</sub>)

The one assumption we can't measure is *s* — is the scalp under the hair
paler than the sun-exposed forehead? Turn the knob and see how much it matters.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    scalp = mo.ui.slider(start=0.85, stop=1.30, step=0.01, value=1.0, label="Scalp albedo relative to forehead (s)", full_width=True)
    guard = mo.ui.dropdown(
        options={"#0.5 (1.5 mm)": 1.5, "#1 (3 mm)": 3.0, "#1.5 (4.5 mm)": 4.5, "#2 (6 mm)": 6.0, "#3 (10 mm)": 10.0, "#4 (13 mm)": 13.0},
        value="#2 (6 mm)",
        label="Assumed clipper guard for the 'before' cut",
    )
    mo.vstack([scalp, guard])
    return guard, scalp


@app.cell(hide_code=True)
def _(ARR, R, guard, mo, np, scalp):
    def joint_unmix(qb, qa, s):
        best = None
        for fb in np.arange(0.5, 1.0001, 0.001):
            h = (qb - s * (1 - fb)) / fb
            if (h < 0).any():
                continue
            d = h - s
            fa = float(np.clip(((qa - s) @ d) / (d @ d), 0, 1))
            err = float(((s * (1 - fa) + fa * h - qa) ** 2).sum())
            if best is None or err < best[0]:
                best = (err, float(fb), fa)
        return best[1], best[2]

    _qb = np.median(ARR["columns_before"][:, 2:], 0)
    _qa = np.median(ARR["columns_after"][:, 2:], 0)
    _fb, _fa = joint_unmix(_qb, _qa, scalp.value)
    _ratio = float(np.log(1 - _fa) / np.log(1 - min(_fb, 0.999)))
    _lb = guard.value
    _la = _ratio * _lb
    # 100k scalp hairs, 70 um shafts, keratin 1.32 g/cm3
    _grams = 100_000 * np.pi * (0.035) ** 2 * (_lb - _la) * 1e-3 * 1.32
    LENGTH_RATIO_LIVE = _ratio
    AFTER_MM_LIVE = _la

    mo.callout(
        mo.md(
            f"With *s* = **{scalp.value:.2f}**: hair hides **{_fb*100:.0f}%** of the scalp before and "
            f"**{_fa*100:.0f}%** after, so the hair is **{(1-_ratio)*100:.0f}% shorter** "
            f"(pipeline default *s* = 1: {R['hair_shorter_pct']['estimate']:.0f}%).\n\n"
            f"If the *before* cut was a **{_lb:g} mm** guard, the *after* cut is **≈{_la:.1f} mm**, "
            f"and Daniel is lighter by **≈{_grams:.1f} g of hair** "
            f"(100,000 scalp hairs × 70 µm shafts × keratin at 1.32 g/cm³)."
        ),
        kind="info",
    )
    return AFTER_MM_LIVE, LENGTH_RATIO_LIVE, joint_unmix


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 3. Extrapolation (satirical, n = 2 time points, please do not cite)")
    years = mo.ui.slider(start=1, stop=15, step=1, value=4, label="Assumed years between photos")
    years
    return (years,)


@app.cell(hide_code=True)
def _(AFTER_MM_LIVE, LENGTH_RATIO_LIVE, mo, np, years):
    _rate = np.log(LENGTH_RATIO_LIVE) / years.value
    _yrs = np.log(0.1 / AFTER_MM_LIVE) / _rate if AFTER_MM_LIVE > 0.1 else 0.0
    mo.callout(
        mo.md(
            f"Assuming hair length decays exponentially (it does not; it is cut), Daniel's "
            f"stubble falls below **0.1 mm** — smoother than an NMR tube — in "
            f"**{_yrs:.1f} years**. His hairline, meanwhile, is projected to stay exactly where "
            f"it is, which is the least funny and most reassuring result in this report."
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(R, mo):
    _g = R["geometry"]
    mo.md(
        f"""
---
## Methods

1. **Landmarks.** Haar-cascade face and eye detection (OpenCV). Each photo is
   rotated so the eyes are level and scaled to a common interpupillary
   distance (IPD), taken as {R['method']['ipd_reference_mm']:.0f} mm (adult mean, Dodgson 2004).
2. **Fiducial.** The forehead scar is found as the strongest thin, dark,
   reddish, near-vertical ridge on the forehead (scar centroid
   {_g['before']['scar_t_ipd']:.3f} vs {_g['after']['scar_t_ipd']:.3f} IPD above the eyes —
   a {(1-R['foreshortening_after_vs_before'])*100:.0f}% foreshortening of the *after* photo).
3. **Hairline.** Per forehead column, the steepest drop in CIELAB chroma
   above the scar. Reported as height relative to the scar; converted to mm
   using the less-foreshortened scar height ({R['scar_height_mm']:.0f} mm).
4. **Hair opacity.** Linear-RGB ratio of the band 4–13 mm behind the
   hairline to the band 4–13 mm in front of it, unmixed jointly across both
   photos with a shared hair albedo.
5. **Length.** Poisson shaft-coverage model; the unknown coverage constant
   cancels in the before/after ratio.
6. **Uncertainty.** {R['method']['n_bootstrap']}-resample bootstrap over
   {R['method']['n_columns'][0]} forehead columns per photo, seed {R['method']['rng_seed']}.

**Limitations.** n = 1 Daniel. Different cameras, lenses and lighting. The
*before* hair is close to opaque, which makes the length ratio's interval
lopsided (opacity is near its ceiling). Absolute lengths require knowing one
clipper guard; only the ratio is measured.

**Erratum.** An earlier version of this report classified stubble pixels as
bare skin and announced an 88% "hair loss" plus a receding hairline. Peer
review (Daniel's friend, who has seen him in person) rejected both claims.
The hairline is fine. The hair is just short.

Reproduce everything with `uv run python -m ocabelododaniel.build`.
"""
    )
    return


if __name__ == "__main__":
    app.run()
