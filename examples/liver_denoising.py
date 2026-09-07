"""Fidelity against task, on the denoising community's own benchmark.

Twelve Siemens liver cases from LDCT-and-Projection-data, vendor reconstructions of both the
routine and the simulated quarter-dose acquisition. A lesion of known size and contrast is
inserted into real parenchyma; the quarter-dose images are denoised at a range of strengths;
and each result is scored two ways - against the full-dose image (fidelity) and against the
task (held-out model observers).

Everything here is measured on real scanner data. The only synthetic element is the lesion,
which is what gives the task a ground truth at all.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from denoiq_core.denoisers import denoise, has_bm3d
from denoiq_core.evaluate import (
    DEFAULT_CONFIG,
    estimate_cho,
    fidelity,
    npwe_estimate,
    prewhitening_estimate,
)

from ldct_io import SIMULATED_DOSE_FRACTION, homogeneous_sites, make_trials, read_image_series

# The prewhitening estimate fits a template over the central roi_size window -- 24 pixels by
# default, so 576 unknowns. With a few hundred trials the estimate is so inefficient that it
# comes out *below* the CHO, which is an upper bound that is not one. The remedy is trials,
# not a smaller window: MAX_PER_CASE is what governs it.
CONFIG = DEFAULT_CONFIG

ROOT = Path(os.environ.get("LDCT_IO_DATA") or r"D:\DevData\TCIA\LDCT-and-Projection-data")
OUTDIR = Path(os.environ.get("LDCT_IO_OUT") or Path(__file__).resolve().parents[1] / "results")
OUTDIR.mkdir(parents=True, exist_ok=True)
CASES = sorted(p.name for p in ROOT.glob("L0*")) + sorted(p.name for p in ROOT.glob("L1*"))
ROI = 48
MAX_PER_CASE = 250  # so a few large livers do not become the whole study
MIN_PER_CASE = 8
LESION_MM, LESION_HU = 8.0, -25.0
LESION_EDGE_SIGMA_MM = 0.5  # so the inserted lesion is no sharper than the scanner
SEED = 0

# The strength parameters of TV, NLM and BM3D are in the units of the image. These images are
# in HU, where the quarter-dose noise is ~24 HU, so a weight tuned for a [0, 1] image does
# nothing at all here -- which is exactly what the first run showed (+0.02 dB).
NOISE_HU = 24.0
DENOISERS = [
    ("none", {}),
    ("gaussian", {"sigma": 0.5}),
    ("gaussian", {"sigma": 0.75}),
    ("gaussian", {"sigma": 1.0}),
    ("gaussian", {"sigma": 1.5}),
    ("gaussian", {"sigma": 2.0}),
    ("gaussian", {"sigma": 3.0}),
    ("tv", {"weight": 0.5 * NOISE_HU}),
    ("tv", {"weight": 1.0 * NOISE_HU}),
    ("tv", {"weight": 2.0 * NOISE_HU}),
    ("nlm", {"h": 0.8 * NOISE_HU}),
    ("nlm", {"h": 1.5 * NOISE_HU}),
]
if has_bm3d():
    DENOISERS += [("bm3d", {"sigma_psd": NOISE_HU})]


def build_trials():
    """Pool trials over cases: low-dose input, full-dose reference, same anatomy and lesion."""
    low_present, low_absent, ref_present, ref_absent = [], [], [], []
    per_case = {}
    signal = None
    spacing = None
    for case in CASES:
        full = read_image_series(ROOT / case / "full_dose_images")
        low = read_image_series(ROOT / case / "low_dose_images")
        if not full.matches(low):
            print(f"  {case}: grids differ, skipped")
            continue
        # Sites chosen on the full-dose volume: the selection must not depend on the noise.
        mid = len(full) // 2
        window = range(max(0, mid - 25), min(len(full), mid + 25))
        try:
            sites = homogeneous_sites(
                full.volume,
                full.spacing,
                ROI,
                hu_range=(0.0, 160.0),
                max_sd=60.0,
                slices=window,
            )
        except ValueError as exc:
            print(f"  {case}: {exc}; skipped")
            continue
        n_class = min(len(sites) // 2, MAX_PER_CASE)
        if n_class < MIN_PER_CASE:
            print(f"  {case}: only {n_class} trials per class available; skipped")
            continue

        common = dict(
            size=ROI,
            diameter_mm=LESION_MM,
            contrast_hu=LESION_HU,
            edge_sigma_mm=LESION_EDGE_SIGMA_MM,
            seed=SEED,
            sites=sites,
            n_trials=n_class,
        )
        t_low = make_trials(low.volume, low.spacing, **common)
        t_ref = make_trials(full.volume, full.spacing, **common)
        low_present.append(t_low.present)
        low_absent.append(t_low.absent)
        ref_present.append(t_ref.present)
        ref_absent.append(t_ref.absent)
        per_case[case] = t_low.meta["n_per_class"]
        signal, spacing = t_low.signal, t_low.spacing
        print(
            f"  {case}: {t_low.meta['n_per_class']} trials per class, "
            f"{t_low.meta['n_sites_found']} sites, {full.kernel}, {full.slice_thickness} mm"
        )
    return (
        np.concatenate(low_present),
        np.concatenate(low_absent),
        np.concatenate(ref_present),
        np.concatenate(ref_absent),
        signal,
        spacing,
        per_case,
    )


def main() -> int:
    print(f"cases: {', '.join(CASES)}")
    print(f"lesion: {LESION_MM} mm at {LESION_HU} HU, edge sigma {LESION_EDGE_SIGMA_MM} mm")
    print(f"low dose = {SIMULATED_DOSE_FRACTION['ABDOMEN']:.0%} of routine (simulated)\n")

    lp, la, rp, ra, signal, spacing, per_case = build_trials()
    n = lp.shape[0]
    print(
        f"\npooled: {n} trials per class, {ROI}x{ROI} at {spacing:.4f} mm "
        f"({ROI * spacing:.1f} mm ROI)"
    )
    noise_full = float(np.std(rp - rp.mean(axis=(1, 2), keepdims=True)))
    noise_low = float(np.std(lp - lp.mean(axis=(1, 2), keepdims=True)))
    print(
        f"ROI spread: full-dose {noise_full:.1f} HU, low-dose {noise_low:.1f} HU "
        f"(ratio {noise_low / noise_full:.2f}; anatomy dominates both)"
    )
    print(f"difference image sd: {float(np.std(lp - rp)):.1f} HU  <- the inserted noise\n")

    rows = []
    for method, params in DENOISERS:
        t0 = time.time()
        label = (
            method if not params else f"{method} " + ",".join(f"{k}={v}" for k, v in params.items())
        )
        dp = denoise(lp, method=method, **params) if method != "none" else lp
        da = denoise(la, method=method, **params) if method != "none" else la

        fid = [
            fidelity(dp[i][None], rp[i], data_range=CONFIG.fidelity_data_range)["psnr"]
            for i in range(0, n, max(1, n // 400))
        ]
        ssim = [
            fidelity(dp[i][None], rp[i], data_range=CONFIG.fidelity_data_range)["ssim"]
            for i in range(0, n, max(1, n // 400))
        ]
        rmse = [
            fidelity(dp[i][None], rp[i], data_range=CONFIG.fidelity_data_range)["rmse"]
            for i in range(0, n, max(1, n // 400))
        ]

        d_ideal = prewhitening_estimate(dp, da, spacing, config=CONFIG)
        d_npwe = npwe_estimate(dp, da, spacing, config=CONFIG)
        d_cho = estimate_cho(dp, da, spacing, config=CONFIG, seed=SEED)

        rows.append(
            dict(
                label=label,
                method=method,
                params=params,
                psnr=float(np.mean(fid)),
                ssim=float(np.mean(ssim)),
                rmse=float(np.mean(rmse)),
                d_ideal=float(d_ideal.d_prime),
                auc_ideal=float(d_ideal.auc),
                d_npwe=float(d_npwe.d_prime),
                d_cho=float(d_cho.d_prime),
            )
        )
        print(
            f"  {label:22s} PSNR {np.mean(fid):6.2f}  SSIM {np.mean(ssim):.4f}  "
            f"RMSE {np.mean(rmse):6.2f}  |  d' ideal {d_ideal.d_prime:5.3f}  "
            f"NPWE {d_npwe.d_prime:5.3f}  CHO {d_cho.d_prime:5.3f}   ({time.time() - t0:.0f}s)",
            flush=True,
        )

    (OUTDIR / "paperB_liver.json").write_text(
        json.dumps({"cases": per_case, "rows": rows}, indent=2)
    )

    raw = rows[0]
    psnr = np.array([r["psnr"] for r in rows])
    print(f"\nrelative to the undenoised quarter-dose input ({raw['label']}):")
    for r in rows[1:]:
        print(
            f"  {r['label']:22s} PSNR {r['psnr'] - raw['psnr']:+6.2f} dB   "
            f"d'_ideal {r['d_ideal'] / raw['d_ideal']:5.2f}x   "
            f"d'_CHO {r['d_cho'] / raw['d_cho']:5.2f}x   "
            f"d'_NPWE {r['d_npwe'] / raw['d_npwe']:5.2f}x"
        )

    gauss = [r for r in rows if r["method"] in ("none", "gaussian")]
    strength = [0.0] + [r["params"]["sigma"] for r in gauss[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(strength, [r["psnr"] for r in gauss], "o-", color="C0")
    axes[0].set_xlabel("Gaussian ----[pixels]")
    axes[0].set_ylabel("PSNR vs full dose [dB]", color="C0")
    ax0b = axes[0].twinx()
    ax0b.plot(strength, [r["ssim"] for r in gauss], "s--", color="C1")
    ax0b.set_ylabel("SSIM", color="C1")
    axes[0].set_title("fidelity")
    axes[0].grid(alpha=0.3)

    for key, marker, label in (
        ("d_ideal", "o-", "held-out prewhitening"),
        ("d_cho", "s-", "CHO"),
        ("d_npwe", "^-", "NPWE"),
    ):
        axes[1].plot(strength, [r[key] for r in gauss], marker, label=label)
    axes[1].set_xlabel("Gaussian ----[pixels]")
    axes[1].set_ylabel(r"$d'$")
    axes[1].set_title("task")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        [r["psnr"] for r in gauss],
        [r["d_ideal"] for r in gauss],
        "o-",
        label="held-out prewhitening",
    )
    axes[2].plot([r["psnr"] for r in gauss], [r["d_cho"] for r in gauss], "s-", label="CHO")
    axes[2].plot([r["psnr"] for r in gauss], [r["d_npwe"] for r in gauss], "^-", label="NPWE")
    axes[2].set_xlabel("PSNR vs full dose [dB]")
    axes[2].set_ylabel(r"$d'$")
    axes[2].set_title("fidelity ----task")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"{len(per_case)} Siemens liver cases, quarter dose, {n} trials per class - "
        f"{LESION_MM:.0f} mm / {LESION_HU:.0f} HU lesion in real parenchyma",
        y=1.02,
    )
    fig.tight_layout()
    out = OUTDIR / "paperB_liver.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}   (PSNR range {psnr.min():.2f}..{psnr.max():.2f} dB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
