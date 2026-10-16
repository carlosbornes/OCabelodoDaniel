"""Reproducible hair-shortening quantification for O Cabelo do Daniel.

Question: how much less hair does Daniel have now? Two separable sub-questions:

1. Did the hairline move (follicle loss)?  Hairline height is measured against
   an anatomical fiducial -- the small forehead scar visible in both photos --
   so head pitch / camera angle (which foreshortens everything above the eyes
   differently in each photo) cancels out.
2. How much shorter is the hair (shearing)?  The hair zone just behind the
   hairline is compared with the forehead skin just in front of it, in linear
   RGB, in the same photo (so illumination cancels). A two-photo linear
   unmixing model estimates hair opacity f; a Poisson shaft-coverage model,
   f = 1 - exp(-k L), turns opacity into a length ratio in which the unknown
   k (density x diameter x viewing geometry, same follicles, same view) cancels.

Uncertainty: bootstrap over forehead columns (B = 2000, fixed seed).

Outputs:
  output/results.json   summary statistics
  output/assets.npz     aligned crops, overlays, profiles, per-column data
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hair.png"
OUT = ROOT / "output"

ADULT_IPD_MM = 63.0  # mean adult interpupillary distance (Dodgson, 2004)
N_BOOTSTRAP = 2000
RNG_SEED = 1988
PAD = 600

# aligned frame: eyes levelled, 400 px per IPD, eye midpoint at (EX, EY)
PX_PER_IPD = 400
EX, EY = 520, 840
FRAME_W, FRAME_H = 1040, 940

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_eyes(gray: np.ndarray, x_range: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(150, 150))
    faces = [f for f in faces if x_range[0] <= f[0] <= x_range[1]]
    if not faces:
        raise RuntimeError(f"no face in x range {x_range}")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    eyes = EYE_CASCADE.detectMultiScale(gray[y : y + h, x : x + w], scaleFactor=1.05, minNeighbors=8)
    # central eye band only: excludes ears, hairline and nose/mouth false positives
    eyes = [
        e for e in eyes
        if h * 0.15 < e[1] + e[3] / 2 < h * 0.55 and w * 0.08 < e[0] + e[2] / 2 < w * 0.92
    ]
    best = None
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            a, b = eyes[i], eyes[j]
            ca = np.array([a[0] + a[2] / 2, a[1] + a[3] / 2])
            cb = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
            if abs(ca[0] - cb[0]) < w * 0.15:
                continue
            level = abs(ca[1] - cb[1])
            if best is None or level < best[0]:
                best = (level, ca, cb)
    if best is None:
        raise RuntimeError("no level eye pair")
    left, right = sorted([best[1], best[2]], key=lambda p: p[0])
    offset = np.array([x, y], dtype=float)
    return left + offset, right + offset


def align(bgr_padded: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rotate so the eyes are level, scale to PX_PER_IPD, centre the eye midpoint."""
    left, right = left + PAD, right + PAD
    mid = (left + right) / 2
    ipd_px = float(np.linalg.norm(right - left))
    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    s = PX_PER_IPD / ipd_px
    rot = cv2.getRotationMatrix2D(tuple(mid), angle, s)
    rot[:, 2] += [EX - mid[0], EY - mid[1]]
    aligned = cv2.warpAffine(bgr_padded, rot, (FRAME_W, FRAME_H), flags=cv2.INTER_CUBIC)
    return aligned, {"ipd_px": ipd_px, "tilt_deg": angle}


def y_of(t: float) -> int:
    """Row in the aligned frame for a height t (in IPD) above the eye line."""
    return int(round(EY - t * PX_PER_IPD))


def to_linear(bgr: np.ndarray) -> np.ndarray:
    v = bgr.astype(np.float64) / 255
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def chroma(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)


def find_head_top(aligned: np.ndarray) -> float:
    lum = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float64), (0, 0), 4)
    ts = np.arange(1.2, 2.05, 0.005)
    band = slice(EX - 40, EX + 41)
    prof = np.array([np.median(lum[y_of(t), band]) for t in ts])
    return float(ts[np.argmax(np.gradient(prof, ts))])


def find_scar(aligned: np.ndarray) -> dict:
    """The forehead scar: a short, thin, darker-and-redder near-vertical line."""
    lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB).astype(np.float64)
    sig = (lab[..., 1] - 128) - 0.35 * (lab[..., 0] - 128)
    sig = cv2.GaussianBlur(sig, (0, 0), 2.4)
    ridge = sig - cv2.GaussianBlur(sig, (0, 0), sigmaX=12, sigmaY=2.4)

    roi = np.zeros(ridge.shape, bool)
    roi[y_of(1.2) : y_of(0.65), EX - 180 : EX + 181] = True
    resp = np.where(roi, ridge, 0)
    mask = (resp > np.percentile(resp[roi], 99.0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        score = area * min(h / max(w, 1), 6)
        if h >= 20 and (best is None or score > best[0]):
            best = (score, i)
    if best is None:
        raise RuntimeError("scar not found")
    ys, xs = np.nonzero(labels == best[1])
    return {
        "t": float((EY - ys.mean()) / PX_PER_IPD),
        "x": float((xs.mean() - EX) / PX_PER_IPD),
        "t_top": float((EY - ys.min()) / PX_PER_IPD),
        "t_bottom": float((EY - ys.max()) / PX_PER_IPD),
        "pixels": (ys, xs),
    }


def trace_hairline(aligned: np.ndarray, scar_t: float, top_t: float, step: int) -> np.ndarray:
    """Per column: hairline = steepest drop in chroma (skin is saturated, hair is not).

    Returns rows of (x_px, hairline_t, hair/skin linear-RGB ratio B, G, R).
    """
    C = cv2.GaussianBlur(chroma(aligned), (0, 0), sigmaX=4, sigmaY=6)
    lin = to_linear(aligned)
    ts = np.arange(scar_t + 0.05, top_t - 0.15, 0.0025)
    rows = []
    for xc in range(EX - 160, EX + 161, step):
        prof = np.array([C[y_of(t), xc] for t in ts])
        th = float(ts[np.argmin(np.gradient(prof, ts))])
        skin = lin[y_of(th - 0.06) : y_of(th - 0.20), xc - 5 : xc + 6].reshape(-1, 3).mean(0)
        hair = lin[y_of(th + 0.20) : y_of(th + 0.06), xc - 5 : xc + 6].reshape(-1, 3).mean(0)
        rows.append((xc, th, *(hair / skin)))
    return np.array(rows)


def joint_unmix(q_before: np.ndarray, q_after: np.ndarray, scalp_albedo: float = 1.0):
    """Fit q = s(1 - f) + f h per channel, h (hair/skin albedo) shared by both photos.

    s is the scalp-vs-forehead skin albedo ratio (1.0 = same skin).
    Returns (f_before, f_after, h).
    """
    best = None
    for fb in np.arange(0.5, 1.0001, 0.001):
        h = (q_before - scalp_albedo * (1 - fb)) / fb
        if (h < 0).any():
            continue
        d = h - scalp_albedo
        fa = float(np.clip(((q_after - scalp_albedo) @ d) / (d @ d), 0, 1))
        err = float(((scalp_albedo * (1 - fa) + fa * h - q_after) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, float(fb), fa, h)
    return best[1], best[2], best[3]


def length_ratio(f_before: float, f_after: float) -> float:
    """Poisson coverage f = 1 - exp(-kL)  =>  L_a / L_b = ln(1 - f_a) / ln(1 - f_b)."""
    f_before = min(f_before, 0.999)
    return float(np.log(1 - f_after) / np.log(1 - f_before))


def profile(aligned: np.ndarray, scar_t: float) -> tuple[np.ndarray, np.ndarray]:
    """Median chroma across the central forehead vs anatomical height (scar units)."""
    C = cv2.GaussianBlur(chroma(aligned), (0, 0), 3)
    u = np.arange(0.6, 1.75, 0.01)
    vals = np.array([np.median(C[y_of(ui * scar_t), EX - 120 : EX + 121]) for ui in u])
    return u, vals


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(DATA))
    if bgr is None:
        raise FileNotFoundError(DATA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(bgr, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    width = bgr.shape[1]

    # left crop = formal headshot (recent) = "after"; right crop = lab photo (earlier) = "before"
    photos = {"before": (width // 2, width), "after": (0, width // 2)}
    per = {}
    for name, xr in photos.items():
        left, right = detect_eyes(gray, xr)
        aligned, geo = align(padded, left, right)
        top_t = find_head_top(aligned)
        scar = find_scar(aligned)
        contour = trace_hairline(aligned, scar["t"], top_t, step=4)
        columns = trace_hairline(aligned, scar["t"], top_t, step=12)  # ~independent columns
        u, prof = profile(aligned, scar["t"])
        per[name] = dict(aligned=aligned, geo=geo, top_t=top_t, scar=scar,
                         contour=contour, columns=columns, profile_u=u, profile=prof)

    b, a = per["before"]["columns"], per["after"]["columns"]
    sb, sa = per["before"]["scar"]["t"], per["after"]["scar"]["t"]

    def estimate(bc: np.ndarray, ac: np.ndarray) -> tuple[float, ...]:
        rb = float(np.median(bc[:, 1]) / sb)
        ra = float(np.median(ac[:, 1]) / sa)
        fb, fa, _ = joint_unmix(np.median(bc[:, 2:], 0), np.median(ac[:, 2:], 0))
        return rb, ra, ra - rb, fb, fa, length_ratio(fb, fa)

    point = estimate(b, a)
    rng = np.random.default_rng(RNG_SEED)
    boots = np.array([
        estimate(b[rng.integers(0, len(b), len(b))], a[rng.integers(0, len(a), len(a))])
        for _ in range(N_BOOTSTRAP)
    ])
    keys = ["hairline_before", "hairline_after", "hairline_shift", "opacity_before", "opacity_after", "length_ratio"]
    stats = {
        k: {"estimate": point[i], "ci95": [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]}
        for i, k in enumerate(keys)
    }

    # scar height in mm: take the less-foreshortened (larger) projection as anatomical
    scar_mm = max(sb, sa) * ADULT_IPD_MM
    shift = stats["hairline_shift"]
    hairline_shift_mm = {"estimate": shift["estimate"] * scar_mm, "ci95": [c * scar_mm for c in shift["ci95"]]}
    lr = stats["length_ratio"]
    shorter_pct = {"estimate": (1 - lr["estimate"]) * 100, "ci95": [(1 - lr["ci95"][1]) * 100, (1 - lr["ci95"][0]) * 100]}

    _, _, h_hat = joint_unmix(np.median(b[:, 2:], 0), np.median(a[:, 2:], 0))
    results = {
        "method": {
            "ipd_reference_mm": ADULT_IPD_MM,
            "n_bootstrap": N_BOOTSTRAP,
            "rng_seed": RNG_SEED,
            "n_columns": [len(b), len(a)],
        },
        "geometry": {
            name: {
                "ipd_px": p["geo"]["ipd_px"],
                "tilt_deg": p["geo"]["tilt_deg"],
                "scar_t_ipd": p["scar"]["t"],
                "scar_x_ipd": p["scar"]["x"],
                "head_top_t_ipd": p["top_t"],
                "hairline_t_ipd": float(np.median(p["columns"][:, 1])),
            }
            for name, p in per.items()
        },
        "foreshortening_after_vs_before": sa / sb,
        "scar_height_mm": scar_mm,
        **stats,
        "hairline_shift_mm": hairline_shift_mm,
        "hair_shorter_pct": shorter_pct,
        "hair_skin_albedo_bgr": h_hat.tolist(),
        "median_band_ratio_bgr": {
            "before": np.median(b[:, 2:], 0).tolist(),
            "after": np.median(a[:, 2:], 0).tolist(),
        },
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # --- assets for the interactive notebook -----------------------------
    k = sb / sa  # vertical stretch that maps "after" anatomy onto "before"
    stretch = np.float32([[1, 0, 0], [0, k, EY * (1 - k)]])
    after_matched = cv2.warpAffine(per["after"]["aligned"], stretch, (FRAME_W, FRAME_H), flags=cv2.INTER_CUBIC)

    def annotate(img: np.ndarray, p: dict, vscale: float = 1.0) -> np.ndarray:
        out = img.copy()
        ys, xs = p["scar"]["pixels"]
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        def vy(y): return int(round(EY - (EY - y) * vscale))
        cv2.rectangle(out, (x0 - 10, vy(y0) - 10), (x1 + 10, vy(y1) + 10), (80, 220, 80), 3)
        ts = p["contour"][:, 1]
        # running median for display only; statistics use the raw per-column values
        ts = np.array([np.median(ts[max(0, i - 3) : i + 4]) for i in range(len(ts))])
        pts = [(int(x), vy(y_of(t))) for x, t in zip(p["contour"][:, 0], ts)]
        for q0, q1 in zip(pts, pts[1:]):
            cv2.line(out, q0, q1, (40, 200, 255), 4, cv2.LINE_AA)
        cv2.line(out, (0, EY), (FRAME_W, EY), (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    np.savez_compressed(
        OUT / "assets.npz",
        before=rgb(per["before"]["aligned"]),
        after=rgb(per["after"]["aligned"]),
        after_matched=rgb(after_matched),
        before_annot=rgb(annotate(per["before"]["aligned"], per["before"])),
        after_annot=rgb(annotate(per["after"]["aligned"], per["after"])),
        after_matched_annot=rgb(annotate(after_matched, per["after"], vscale=k)),
        profile_u=per["before"]["profile_u"],
        profile_before=per["before"]["profile"],
        profile_after=per["after"]["profile"],
        columns_before=b,
        columns_after=a,
    )
    print(json.dumps({k: v for k, v in results.items() if k != "geometry"}, indent=2))


if __name__ == "__main__":
    main()
