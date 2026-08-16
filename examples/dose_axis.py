"""Does any of it survive a different operating point? Liver against chest.

Every number in the liver study rests on one acquisition, and the obvious question about a
claim like "no processing raises detectability" is whether it survives a different one. This
collection supplies a very different one without any new machinery.

It is *not* a clean dose axis, and saying so matters. The chest cases were simulated at 10 %
of the routine dose against the abdomen's 25 %, but they were also reconstructed with a sharp
kernel instead of a smooth one (B50f against B30f) at 1.5 mm instead of 5 mm and on a 0.78 mm
pixel instead of 0.625 mm. All four push the same way, and the measured noise comes out 8.2
times larger -- 208 HU against 25 HU -- most of that from the kernel and the slice thickness
rather than from the dose. The denoiser strengths here are therefore set in millimetres and
in units of the measured noise, never in pixels, so that the same filter means the same thing
in both arms.

The prediction, and what happened
---------------------------------
Written before the run: *the ratio of every processed result to its own closed-form ceiling
should be about the same in both arms, because the data-processing argument does not know
what the dose, the kernel or the slice thickness were.*

Half of that was right and half of it was a confusion, and the measurement separates them.
The bound held everywhere -- nothing in either arm exceeded its ceiling, the largest being
the unprocessed input at 0.77 -- and the two unprocessed inputs sit at a similar fraction of
their own ceilings, 0.77 against 0.67. But the *processed* ratios do not match at all:

    strength              liver    chest
    none                   0.77     0.67
    gaussian 0.75 mm       0.70     0.21
    gaussian 1.00 mm       0.67     0.11
    tv 2x noise            0.68     0.22

The error was to read the data-processing inequality as a statement about how much a filter
costs. It is not; it says only that nothing can be gained. How much is *lost* depends on
where the noise sits relative to the signal, and that is a property of the reconstruction.
The chest's sharp kernel puts its noise at high frequencies that a prewhitening observer was
already discounting, so smoothing there removes noise the observer was not fighting while
attenuating the lesion regardless: nearly all cost, no benefit. The liver's smooth kernel
puts more of its noise where the signal is, so the same physical smoothing costs less.

Which makes the fidelity-task divergence far worse in the chest than in the liver: +10.1 dB
of PSNR for a 91 % loss of detectability, against +4.3 dB for 21 %.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from denoiq_core.denoisers import denoise
from denoiq_core.evaluate import fidelity
from taskiq_core import ideal_linear, nps_2d

from ldct_io import (
    SIMULATED_DOSE_FRACTION,
    homogeneous_sites,
    make_paired_trials,
    noise_only,
    paired_d_prime,
    read_image_series,
)

ROOT = Path(r"D:\DevData\TCIA\LDCT-and-Projection-data")
OUTDIR = Path(os.environ.get("LDCT_IO_OUT") or Path(__file__).resolve().parents[1] / "results")
OUTDIR.mkdir(parents=True, exist_ok=True)
ROI, MAX_PER_CASE, SEED, N_FOLDS = 48, 250, 0, 5
LESION_MM, LESION_HU, LESION_EDGE_SIGMA_MM = 8.0, -25.0, 0.5

#: Selection criteria per body part. The chest has far less soft tissue in a 30 mm window
#: than a liver does, so the window on HU is wider and the tolerance on spread larger; the
#: task itself — a low-contrast lesion in soft tissue — is deliberately kept the same, so that
#: what changes between the two arms is the dose and not the question.
REGIONS: dict[str, dict[str, Any]] = {
    "liver": dict(
        glob="L0*",
        dose=SIMULATED_DOSE_FRACTION["ABDOMEN"],
        criteria=dict(hu_range=(0.0, 160.0), max_sd=60.0),
    ),
    "chest": dict(
        glob="C0*",
        dose=SIMULATED_DOSE_FRACTION["CHEST"],
        criteria=dict(hu_range=(-40.0, 160.0), max_sd=90.0),
    ),
}

NOISE_SCALE = {"liver": 25.0, "chest": 208.0}  # measured, sets the strengths that scale with it

#: Denoiser strengths in *physical* units, because the two arms are not on the same grid --
#: 0.625 mm against 0.78 mm. A Gaussian of "sigma = 2 pixels" blurs 25 % more of a lesion in
#: one arm than the other, and comparing the two would be comparing two different filters.
#: TV and NLM take their strength in the units of the image, so they scale with the noise.
STRENGTHS: list[tuple[str, Any]] = [
    ("none", lambda noise, mm: {}),
    ("gaussian", lambda noise, mm: {"sigma": 0.5 / mm}),
    ("gaussian", lambda noise, mm: {"sigma": 0.75 / mm}),
    ("gaussian", lambda noise, mm: {"sigma": 1.0 / mm}),
    ("gaussian", lambda noise, mm: {"sigma": 1.5 / mm}),
    ("tv", lambda noise, mm: {"weight": 1.0 * noise}),
    ("tv", lambda noise, mm: {"weight": 2.0 * noise}),
    ("nlm", lambda noise, mm: {"h": 0.8 * noise}),
]
#: Human-readable name of each strength, the same in both arms so the two can be lined up.
STRENGTH_LABELS = [
    "none",
    "gaussian 0.50 mm",
    "gaussian 0.75 mm",
    "gaussian 1.00 mm",
    "gaussian 1.50 mm",
    "tv 1x noise",
    "tv 2x noise",
    "nlm 0.8x noise",
]


def invertible(grid: np.ndarray, ridge: float = 1e-2, floor: float = 1e-3) -> np.ndarray:
    """Fill the zeroed DC bin, add a ridge, apply a floor."""
    out = np.array(grid, dtype=np.float64, copy=True)
    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    out[cy, cx] = float(
        np.mean([out[cy - 1, cx], out[cy + 1, cx], out[cy, cx - 1], out[cy, cx + 1]])
    )
    out = out + ridge * float(out.mean())
    return np.maximum(out, floor * float(out.max()))


def cross_fitted_paired(present: np.ndarray, absent: np.ndarray, spacing: float) -> float:
    """Held-out prewhitening d' on paired trials."""
    n = present.shape[0]
    fold = np.arange(n) % N_FOLDS
    sp, sa = np.empty(n), np.empty(n)
    for k in range(N_FOLDS):
        train, test = fold != k, fold == k
        signal_hat = present[train].mean(axis=0) - absent[train].mean(axis=0)
        diffs = present[train] - absent[train]
        noise_nps = nps_2d(diffs - diffs.mean(axis=0), spacing, detrend="mean")
        template = np.real(
            np.fft.ifft2(np.fft.fft2(signal_hat) / np.fft.ifftshift(invertible(noise_nps.nps)))
        )
        sp[test] = (present[test] * template).sum(axis=(1, 2))
        sa[test] = (absent[test] * template).sum(axis=(1, 2))
    return paired_d_prime(sp, sa)


def gather(region: str) -> dict[str, Any]:
    """Paired trials, the reference (full-dose) trials, and the noise patches, over a region."""
    spec = REGIONS[region]
    present, absent, reference, patches, used = [], [], [], [], {}
    signal = spacing = None
    for path in sorted(ROOT.glob(spec["glob"])):
        case = path.name
        try:
            full = read_image_series(path / "full_dose_images")
            low = read_image_series(path / "low_dose_images")
        except (ValueError, FileNotFoundError):
            continue
        if not full.matches(low):
            continue
        noise, _ = noise_only(full, low, dose_fraction=spec["dose"])
        mid = len(full) // 2
        window = range(max(0, mid - 25), min(len(full), mid + 25))
        try:
            sites = homogeneous_sites(
                full.volume, full.spacing, ROI, slices=window, **spec["criteria"]
            )
        except ValueError:
            print(f"  {case}: no usable site")
            continue
        n_take = min(len(sites), MAX_PER_CASE)
        if n_take < 16:
            continue
        common = dict(
            size=ROI,
            diameter_mm=LESION_MM,
            contrast_hu=LESION_HU,
            edge_sigma_mm=LESION_EDGE_SIGMA_MM,
            n_trials=n_take,
            seed=SEED,
        )
        trials = make_paired_trials(full.volume, noise, full.spacing, sites, **common)
        clean = make_paired_trials(full.volume, np.zeros_like(noise), full.spacing, sites, **common)
        half = ROI // 2
        patches.append(
            np.stack([noise[s, r - half : r + half, c - half : c + half] for s, r, c in sites])
        )
        present.append(trials.present)
        absent.append(trials.absent)
        reference.append(clean.present)
        used[case] = n_take
        signal, spacing = trials.signal, trials.spacing
    if not used:
        raise ValueError(f"no usable case for {region}")
    print(f"  {region}: {sum(used.values())} pairs from {len(used)} cases")
    return dict(
        present=np.concatenate(present),
        absent=np.concatenate(absent),
        reference=np.concatenate(reference),
        patches=np.concatenate(patches).astype(np.float64),
        signal=signal,
        spacing=spacing,
        used=used,
        dose=spec["dose"],
    )


def analyse(region: str) -> dict[str, Any]:
    data = gather(region)
    spacing, signal = data["spacing"], data["signal"]
    noise_nps = nps_2d(data["patches"], spacing, detrend="mean")
    noise_sd = float(np.sqrt(noise_nps.integral))
    ceiling = ideal_linear(
        signal, invertible(noise_nps.nps), spacing, nps_layout="centered"
    ).d_prime
    print(f"  noise sd {noise_sd:.1f} HU, closed-form ceiling d' = {ceiling:.3f}")

    rows = []
    scale = NOISE_SCALE[region]
    for (method, make), label in zip(STRENGTHS, STRENGTH_LABELS, strict=True):
        params = make(scale, spacing)
        dp = (
            data["present"]
            if method == "none"
            else denoise(data["present"], method=method, **params)
        )
        da = (
            data["absent"] if method == "none" else denoise(data["absent"], method=method, **params)
        )
        d = cross_fitted_paired(dp, da, spacing)
        step = max(1, dp.shape[0] // 300)
        psnr = float(
            np.mean(
                [
                    fidelity(dp[i][None], data["reference"][i], data_range=400.0)["psnr"]
                    for i in range(0, dp.shape[0], step)
                ]
            )
        )
        rows.append(dict(label=label, d_prime=float(d), ratio=float(d / ceiling), psnr=psnr))
        print(
            f"    {label:16s} d' {d:6.3f}  {d / ceiling:5.2f}x ceiling   PSNR {psnr:6.2f} dB",
            flush=True,
        )
    return dict(
        region=region,
        dose=data["dose"],
        noise_sd=noise_sd,
        ceiling=ceiling,
        rows=rows,
        cases=data["used"],
        n_pairs=int(data["present"].shape[0]),
    )


def main() -> int:
    results = {}
    for region in REGIONS:
        print(f"\n=== {region} ({REGIONS[region]['dose']:.0%} of routine dose) ===")
        try:
            results[region] = analyse(region)
        except ValueError as exc:
            print(f"  skipped: {exc}")

    if len(results) == 2:
        liver, chest = results["liver"], results["chest"]
        from_dose = np.sqrt((1 / chest["dose"] - 1) / (1 / liver["dose"] - 1))
        measured = chest["noise_sd"] / liver["noise_sd"]
        print("\n=== the two operating points ===")
        print(
            f"  noise sd          liver {liver['noise_sd']:.1f} HU   chest "
            f"{chest['noise_sd']:.1f} HU   ratio {measured:.2f}"
        )
        print(
            f"    the dose alone would give {from_dose:.2f}; the rest is the sharp kernel, "
            f"the thinner slice and the larger pixel"
        )
        print(
            f"  ceiling d'        liver {liver['ceiling']:.2f}       chest "
            f"{chest['ceiling']:.2f}       ratio "
            f"{chest['ceiling'] / liver['ceiling']:.2f}"
        )
        print("\n  ratio to own ceiling, by denoiser:")
        for a, b in zip(liver["rows"], chest["rows"], strict=True):
            print(
                f"    {a['label']:16s} liver {a['ratio']:5.2f}   chest {b['ratio']:5.2f}   "
                f"difference {b['ratio'] - a['ratio']:+.3f}"
            )
        spread = max(
            abs(b["ratio"] - a["ratio"]) for a, b in zip(liver["rows"], chest["rows"], strict=True)
        )
        print(f"\n  largest difference in ratio-to-ceiling across a 2.5x dose change: {spread:.3f}")

    (OUTDIR / "dose_axis.json").write_text(json.dumps(results, indent=2, default=float))

    if len(results) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for region, marker in (("liver", "o-"), ("chest", "s-")):
            r = results[region]
            labels = [row["label"] for row in r["rows"]]
            axes[0].plot(
                [row["d_prime"] for row in r["rows"]],
                marker,
                label=f"{region}, {r['dose']:.0%} dose",
            )
            axes[0].axhline(r["ceiling"], ls="--", lw=1, color="C0" if region == "liver" else "C1")
            axes[1].plot([row["ratio"] for row in r["rows"]], marker, label=region)
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel(r"$d'$   (dashed: own closed-form ceiling)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel(r"$d'$ / own ceiling")
        axes[1].axhline(1.0, color="C3", lw=1.5)
        axes[1].set_ylim(0, 1.1)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].set_title("the same fraction of the ceiling at both doses?")
        fig.tight_layout()
        out = OUTDIR / "dose_axis.png"
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
