"""A ceiling that survives real anatomy, and what processing does under it.

The prewhitening ceiling that works on a uniform phantom does not transfer to a liver: the
variation an observer must see through is mostly anatomy, which is neither stationary nor
Gaussian, and a template estimated against it comes out below a channelised observer. This
example rebuilds the ceiling on the one thing this collection makes measurable — the noise
that dose reduction actually costs, on real anatomy, with the anatomy cancelled.

Construction
------------
The low-dose series is the same projections with noise inserted, so ``low - full`` is a
measured realisation of that noise. Holding one background fixed and drawing two independent
noise patches for the two classes gives a task whose only stochastic component is the noise:
background-known-exactly, paired. On the unprocessed input the ideal linear observer is then
available in closed form and carries no sampling error. On processed images it is estimated,
cross-fitted, and can only be compared with that closed form.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from denoiq_core.denoisers import denoise
from taskiq_core import burgess_eye_filter, ideal_linear, nps_2d, npwe

from ldct_io import (
    SIMULATED_DOSE_FRACTION,
    homogeneous_sites,
    make_paired_trials,
    noise_only,
    paired_d_prime,
    read_image_series,
)

ROOT = Path(r"D:\DevData\TCIA\LDCT-and-Projection-data")
OUTDIR = Path(
    r"C:\Users\YAMAMO~1\AppData\Local\Temp\claude\D--DevGit-DICOM-Viewer"
    r"\4b28a974-1910-45c1-aa13-1fed27b3f70c\scratchpad"
)
CASES = sorted(p.name for p in ROOT.glob("L0*"))
ROI, MAX_PER_CASE = 48, 250
LESION_MM, LESION_HU, LESION_EDGE_SIGMA_MM = 8.0, -25.0, 0.5
SEED, N_FOLDS = 0, 5

NOISE_HU = 25.0
DENOISERS = [
    ("none", {}),
    ("gaussian", {"sigma": 0.75}),
    ("gaussian", {"sigma": 1.0}),
    ("gaussian", {"sigma": 1.5}),
    ("gaussian", {"sigma": 2.0}),
    ("tv", {"weight": 1.0 * NOISE_HU}),
    ("tv", {"weight": 2.0 * NOISE_HU}),
    ("nlm", {"h": 0.8 * NOISE_HU}),
]


def invertible(grid: np.ndarray, ridge: float = 1e-2, floor: float = 1e-3) -> np.ndarray:
    """Fill the zeroed DC bin, add a ridge, apply a floor — see denoiq-core."""
    out = np.array(grid, dtype=np.float64, copy=True)
    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    out[cy, cx] = float(
        np.mean([out[cy - 1, cx], out[cy + 1, cx], out[cy, cx - 1], out[cy, cx + 1]])
    )
    out = out + ridge * float(out.mean())
    return np.maximum(out, floor * float(out.max()))


def cross_fitted_paired(present: np.ndarray, absent: np.ndarray, spacing: float) -> float:
    """Held-out prewhitening d' on paired trials: template from other folds, scores from this.

    The template is the estimated effective signal divided by the estimated noise power, and
    both come only from the training folds — so no image is ever scored by a template that
    saw it. On paired trials the noise the template must invert is estimated from the *pair
    differences*, which is the one place the anatomy is guaranteed absent.
    """
    n = present.shape[0]
    fold = np.arange(n) % N_FOLDS
    scores_p, scores_a = np.empty(n), np.empty(n)
    for k in range(N_FOLDS):
        train, test = fold != k, fold == k
        signal_hat = present[train].mean(axis=0) - absent[train].mean(axis=0)
        differences = present[train] - absent[train]
        noise_nps = nps_2d(differences - differences.mean(axis=0), spacing, detrend="mean")
        template = np.real(
            np.fft.ifft2(np.fft.fft2(signal_hat) / np.fft.ifftshift(invertible(noise_nps.nps)))
        )
        scores_p[test] = (present[test] * template).sum(axis=(1, 2))
        scores_a[test] = (absent[test] * template).sum(axis=(1, 2))
    return paired_d_prime(
        scores_p[np.argsort(fold, kind="stable")], scores_a[np.argsort(fold, kind="stable")]
    )


def main() -> int:
    print(f"low dose = {SIMULATED_DOSE_FRACTION['ABDOMEN']:.0%} of routine (simulated)")
    present, absent, noise_patches = [], [], []
    signal = None
    spacing = None
    used = {}
    for case in CASES:
        full = read_image_series(ROOT / case / "full_dose_images")
        low = read_image_series(ROOT / case / "low_dose_images")
        if not full.matches(low):
            continue
        noise, factor = noise_only(full, low, dose_fraction=SIMULATED_DOSE_FRACTION["ABDOMEN"])
        mid = len(full) // 2
        window = range(max(0, mid - 25), min(len(full), mid + 25))
        try:
            sites = homogeneous_sites(
                full.volume, full.spacing, ROI, hu_range=(0.0, 160.0), max_sd=60.0, slices=window
            )
        except ValueError:
            continue
        n_take = min(len(sites), MAX_PER_CASE)
        if n_take < 16:
            continue
        trials = make_paired_trials(
            full.volume,
            noise,
            full.spacing,
            sites,
            size=ROI,
            diameter_mm=LESION_MM,
            contrast_hu=LESION_HU,
            edge_sigma_mm=LESION_EDGE_SIGMA_MM,
            n_trials=n_take,
            seed=SEED,
        )
        half = ROI // 2
        noise_patches.append(
            np.stack([noise[s, r - half : r + half, c - half : c + half] for s, r, c in sites])
        )
        present.append(trials.present)
        absent.append(trials.absent)
        signal, spacing = trials.signal, trials.spacing
        used[case] = n_take
        print(
            f"  {case}: {n_take} pairs, noise sd {noise.std():.1f} HU "
            f"(full-dose NPS = this / {factor:.0f})"
        )

    present = np.concatenate(present)
    absent = np.concatenate(absent)
    noise_patches = np.concatenate(noise_patches).astype(np.float64)
    n = present.shape[0]
    print(
        f"\n{n} pairs from {len(used)} cases; {noise_patches.shape[0]} noise patches "
        f"for the spectrum"
    )

    # --- the ceiling, in closed form on the unprocessed input -------------------------------
    noise_nps = nps_2d(noise_patches, spacing, detrend="mean")
    print(
        f"measured noise: sd {np.sqrt(noise_nps.integral):.1f} HU, "
        f"NPS peak at {noise_nps.frequency[np.argmax(noise_nps.nps_radial)]:.3f} cyc/mm"
    )
    ceiling = ideal_linear(signal, invertible(noise_nps.nps), spacing, nps_layout="centered")
    ceiling_npwe = npwe(
        signal, invertible(noise_nps.nps), spacing, eye_filter=burgess_eye_filter(0.25)
    )
    print(f"\nclosed-form ceiling (ideal, BKE): d' = {ceiling.d_prime:.3f}")
    print(f"closed-form NPWE + eye filter    : d' = {ceiling_npwe.d_prime:.3f}")

    rows = []
    for method, params in DENOISERS:
        t0 = time.time()
        label = (
            method
            if not params
            else f"{method} " + ",".join(f"{k}={v:g}" for k, v in params.items())
        )
        dp = present if method == "none" else denoise(present, method=method, **params)
        da = absent if method == "none" else denoise(absent, method=method, **params)
        d = cross_fitted_paired(dp, da, spacing)
        rows.append(dict(label=label, d_prime=float(d), ratio=float(d / ceiling.d_prime)))
        print(
            f"  {label:16s} d' = {d:6.3f}   {d / ceiling.d_prime:5.2f} x the ceiling"
            f"   ({time.time() - t0:.0f}s)",
            flush=True,
        )

    violations = [r for r in rows if r["ratio"] > 1.02]
    print(
        f"\nabove the ceiling: {len(violations)} of {len(rows)}"
        + (f" -> {[r['label'] for r in violations]}" if violations else "")
    )

    (OUTDIR / "liver_ceiling.json").write_text(
        json.dumps({"cases": used, "ceiling": ceiling.d_prime, "rows": rows}, indent=2)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(rows))
    ax.barh(y, [r["d_prime"] for r in rows], color="C0")
    ax.axvline(
        ceiling.d_prime,
        color="C3",
        lw=2,
        label=f"closed-form ceiling, unprocessed input ({ceiling.d_prime:.2f})",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"held-out $d'$, background known exactly")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    ax.set_title(
        f"{n} paired trials, {len(used)} liver cases, quarter dose\n"
        f"{LESION_MM:.0f} mm / {LESION_HU:.0f} HU lesion in real parenchyma"
    )
    fig.tight_layout()
    out = OUTDIR / "liver_ceiling.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
