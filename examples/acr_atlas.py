"""The framework, unchanged, on a real scanner: MTF -> NPS -> NEQ -> detectability.

The reconstruction kernel is swept. Nothing about the acquisition is simulated: the same
measured projections are reconstructed with progressively smoother filters, which moves MTF
and NPS together exactly as changing a scanner's kernel does, and the physical-to-task
transfer is read off the result.

Physical metrics come from ldct_io on the ACR phantom; NPS, NEQ and the model observers come
from taskiq_core with no modification.
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
from taskiq_core import burgess_eye_filter, fit_transfer, ideal_linear, neq, nps_2d, npwe

from ldct_io import (
    fan_beam_fbp,
    fit_edge_circle,
    index_series,
    radial_mtf,
    single_slice_rebin,
    to_hu,
)

DATA = Path(
    os.environ.get("LDCT_IO_DATA")
    or r"D:\DevData\TCIA\LDCT-and-Projection-data\ACR_Phantom\projections"
)
OUTDIR = Path(os.environ.get("LDCT_IO_OUT") or Path(__file__).resolve().parents[1] / "results")
OUTDIR.mkdir(parents=True, exist_ok=True)
Z, FOV, NPIX = -168.0, 260.0, 512
ROI = 64
# The ACR low-contrast module's own task: a 6 HU cylinder. 5 mm across.
LESION_DIAMETER_MM, LESION_CONTRAST_HU = 5.0, 6.0
KERNELS = [
    ("none", 1.0),
    ("hann", 1.0),
    ("hann", 0.8),
    ("hann", 0.6),
    ("hann", 0.45),
    ("hann", 0.35),
    ("hann", 0.25),
]


def uniform_rois(hu, spacing, centre, radius_limit, size):
    n = hu.shape[0]
    axis = (np.arange(n) - (n - 1) / 2.0) * spacing
    half = size // 2
    out = []
    for iy in range(half, n - half, half):
        for ix in range(half, n - half, half):
            dx, dy = axis[ix] - centre[0], axis[iy] - centre[1]
            if np.hypot(dx, dy) + half * spacing * np.sqrt(2) < radius_limit:
                out.append(hu[iy - half : iy + half, ix - half : ix + half])
    return np.stack(out)


def invertible_nps(nps, ridge=1e-2, floor=1e-3):
    """A measured NPS a prewhitening observer can safely divide by.

    Three things, each for a stated reason (the same treatment denoiq-core applies):

    1. The DC bin is filled from its neighbours. ``nps_2d`` detrends, which *zeroes* DC by
       construction — that is not a measurement of no noise. Left at zero, ``1/NPS`` puts
       essentially infinite weight on the one frequency where a disk has most of its power.
    2. A ridge is added, bounding the inverse where the measured power has fluctuated to
       nothing — which, after apodisation, is most of the frequency plane.
    3. A floor keeps the dynamic range inside what a prewhitening observer can be trusted
       with at all.

    These stabilise the estimate and may cost it efficiency; they are not a one-sided bound.
    """
    grid = np.array(nps.nps, dtype=np.float64, copy=True)
    cy, cx = grid.shape[0] // 2, grid.shape[1] // 2
    grid[cy, cx] = float(
        np.mean([grid[cy - 1, cx], grid[cy + 1, cx], grid[cy, cx - 1], grid[cy, cx + 1]])
    )
    grid = grid + ridge * float(grid.mean())
    return np.maximum(grid, floor * float(grid.max()))


def band_limit(freq, mtf, level=0.05):
    """Truncate a measured MTF where it stops being a measurement.

    Above the frequency at which the MTF has fallen to a few per cent it is estimation noise,
    not transfer: the ESF's own noise puts a floor of a few per cent on |FFT(LSF)|. That tail
    is harmless in the MTF curve itself and ruinous downstream, because NEQ and a prewhitening
    observer both *divide* by an NPS that the same apodisation has driven to nearly nothing.
    A few per cent of spurious MTF over a large, nearly noiseless region of the frequency
    plane then dominates the integral — and the resulting d' is large, smooth and entirely
    wrong. Cutting the tail is not cosmetic; without it the numbers mean nothing.
    """
    below = np.flatnonzero(mtf < level)
    f_max = float(freq[below[0]]) if below.size else float(freq[-1])
    out = np.where(freq <= f_max, mtf, 0.0)
    return out, f_max


def blurred_disk(size, spacing, diameter_mm, contrast, freq, mtf):
    """An ideal disk imaged through the *measured* MTF."""
    axis = (np.arange(size) - (size - 1) / 2.0) * spacing
    X, Y = np.meshgrid(axis, axis)
    # Supersample the disk so its own edge is not a one-pixel staircase.
    sub = 4
    fine = (np.arange(size * sub) - (size * sub - 1) / 2.0) * (spacing / sub)
    XF, YF = np.meshgrid(fine, fine)
    ideal = (np.hypot(XF, YF) <= diameter_mm / 2.0).astype(np.float64)
    ideal = ideal.reshape(size, sub, size, sub).mean((1, 3)) * contrast

    fx = np.fft.fftfreq(size, spacing)
    FX, FY = np.meshgrid(fx, fx)
    fr = np.hypot(FX, FY)
    transfer = np.interp(fr, freq, mtf, left=mtf[0], right=0.0)
    return np.real(np.fft.ifft2(np.fft.fft2(ideal) * transfer)), X, Y


def _dynamic_range_excluding_dc(plane: np.ndarray) -> float:
    """max/min of a centred NPS plane, with DC removed.

    Mean- or polynomial-detrending drives the DC bin to ~1e-31 by construction. That
    is bookkeeping, not a decayed spectrum, and reading it as one makes the check fire
    on every field including white noise.
    """
    q = np.asarray(plane, dtype=float).copy()
    q[q.shape[0] // 2, q.shape[1] // 2] = np.nan
    positive = q[np.isfinite(q) & (q > 0)]
    return float(np.nanmax(q) / positive.min()) if positive.size else float("inf")


def main() -> int:
    series = index_series(DATA)
    g = series.geometry
    sino = single_slice_rebin(series, Z, focal_spot_class=0)
    print(f"ACR uniform module, z = {Z} mm: {sino.meta['n_views']} views")

    rows = []
    curves = {}
    for apod, cutoff in KERNELS:
        t0 = time.time()
        recon = fan_beam_fbp(
            sino.sinogram,
            g,
            sino.angles,
            fov=FOV,
            n_pixels=NPIX,
            apodisation=apod,
            cutoff=cutoff,
        )
        hu = to_hu(recon.image, g.water_attenuation)
        sp = recon.spacing
        label = "ramp" if apod == "none" else f"hann {cutoff:.2f}"

        circle = fit_edge_circle(hu, sp, search_range=(85.0, 112.0))
        mtf = radial_mtf(hu, sp, circle, band=6.0, background="asymptote")

        rois = uniform_rois(hu, sp, (circle.centre_x, circle.centre_y), 72.0, ROI)
        nps = nps_2d(rois, sp, detrend="poly2")

        mtf_bl, f_max = band_limit(mtf.frequency, mtf.mtf)
        band = mtf.frequency <= f_max
        neq_res = neq(np.vstack([mtf.frequency[band], mtf_bl[band]]), nps, nps_floor_fraction=1e-3)

        signal, _, _ = blurred_disk(
            ROI, sp, LESION_DIAMETER_MM, LESION_CONTRAST_HU, mtf.frequency, mtf_bl
        )
        d_ideal = ideal_linear(signal, invertible_nps(nps), sp, nps_layout="centered").d_prime
        d_npwe = npwe(signal, nps, sp, eye_filter=burgess_eye_filter(1.0)).d_prime

        rows.append(
            dict(
                label=label,
                apodisation=apod,
                cutoff=cutoff,
                mtf50=mtf.mtf50,
                mtf10=mtf.mtf10,
                f_max=f_max,
                noise_sd=float(np.sqrt(nps.integral)),
                nps_peak_f=float(nps.frequency[np.argmax(nps.nps_radial)]),
                neq_peak=float(neq_res.peak),
                neq_integral=float(neq_res.integral),
                d_ideal=float(d_ideal),
                d_npwe=float(d_npwe),
                efficiency=float((d_npwe / d_ideal) ** 2),
                # The internal identities, evaluated on measured data. These need no
                # ground truth, which is the whole reason they can be run here at all:
                # the true MTF and NPS of a clinical scanner are not known in closed form.
                parseval_residual=float(abs(nps.integral / nps.variance - 1.0)),
                nps_dynamic_range=_dynamic_range_excluding_dc(nps.nps),
                signal_area_residual=float(
                    abs(
                        signal.sum()
                        * sp
                        * sp
                        / (LESION_CONTRAST_HU * np.pi * (LESION_DIAMETER_MM / 2.0) ** 2)
                        - 1.0
                    )
                ),
            )
        )
        curves[label] = (
            mtf.frequency,
            mtf.mtf,
            nps.frequency,
            nps.nps_radial,
            neq_res.frequency,
            neq_res.neq,
        )
        print(
            f"  {label:11s} MTF50 {mtf.mtf50:.3f}  f_max {f_max:.3f}  "
            f"noise {np.sqrt(nps.integral):6.1f} HU  NEQ_int {neq_res.integral:.3e}  "
            f"d'_ideal {d_ideal:6.3f}  d'_NPWE {d_npwe:6.3f}  "
            f"| Parseval {rows[-1]['parseval_residual']:.1e}  "
            f"NPS range {rows[-1]['nps_dynamic_range']:.1e}  "
            f"area {rows[-1]['signal_area_residual']:.1e}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # The transfer the study is about: does the physics predict the task?
    neq_int = np.array([r["neq_integral"] for r in rows])
    for obs in ("d_ideal", "d_npwe"):
        y = np.array([r[obs] for r in rows]) ** 2
        fit = fit_transfer(neq_int, y, names=["NEQ integral"], fit_intercept=False)
        print(f"\n{obs}^2 = {fit.coef[0]:.4g} * NEQ_integral    R^2 = {fit.r_squared:.4f}")

    (OUTDIR / "acr_atlas.json").write_text(json.dumps(rows, indent=2))

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 300})
    fig, axgrid = plt.subplots(2, 2, figsize=(7.8, 6.8))
    axes = axgrid.ravel()
    for label, (f, m, nf, npsr, qf, q) in curves.items():
        axes[0].plot(f, m, lw=1.3, label=label)
        axes[1].plot(nf, npsr, lw=1.3)
        axes[2].plot(qf, q, lw=1.3)
    axes[0].set_xlabel("cycles/mm")
    axes[0].set_ylabel("MTF")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9.5)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("measured MTF (ACR outer edge)")
    axes[1].set_xlabel("cycles/mm")
    axes[1].set_ylabel(r"NPS [HU$^2$mm$^2$]")
    axes[1].set_xlim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("measured NPS (uniform module)")
    axes[2].set_xlabel("cycles/mm")
    axes[2].set_ylabel(r"NEQ [(HU$^2$mm$^2$)$^{-1}$]")
    axes[2].set_xlim(0, 1)
    axes[2].grid(alpha=0.3)
    axes[2].set_yscale("log")
    axes[2].set_title("NEQ")

    mtf50 = np.array([r["mtf50"] for r in rows])
    d_i = np.array([r["d_ideal"] for r in rows])
    d_n = np.array([r["d_npwe"] for r in rows])
    noise = np.array([r["noise_sd"] for r in rows])
    intact = np.array([r["f_max"] for r in rows]) >= 0.33

    axes[3].plot(mtf50, d_i, "o-", label="ideal (prewhitening)")
    axes[3].plot(mtf50, d_n, "s-", label="NPWE + eye filter")
    axes[3].plot(mtf50[intact], d_i[intact], "o", ms=11, mfc="none", mec="C0", label="band intact")
    # Only the two ends are labelled. At a type size that survives reduction to a text
    # column, seven annotations collide with each other and with the legend; the ends
    # carry the range, and Table 2 of the manuscript carries all seven exactly.
    # rows[0] is the ramp, which is the *rightmost* point (highest MTF50), and the
    # last row is hann 0.25 at the left. Each label is pushed away from its own edge.
    ends = ((0, -8, -14, "right", "top"), (len(rows) - 1, 9, 10, "left", "bottom"))
    for idx, dx, dy, ha, va in ends:
        r = rows[idx]
        axes[3].annotate(
            f"{r['label']}, {r['noise_sd']:.0f} HU",
            (mtf50[idx], d_n[idx]),
            fontsize=9.5,
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
        )
    axes[3].set_xlabel("MTF50 [cycles/mm]  ← smoother kernel")
    axes[3].set_ylabel(r"$d'$")
    axes[3].set_ylim(0, None)
    axes[3].margins(x=0.10)
    axes[3].grid(alpha=0.3)
    axes[3].legend(fontsize=9.5, loc="lower left", framealpha=0.95)
    axes[3].set_title(
        f"{LESION_DIAMETER_MM:.0f} mm / {LESION_CONTRAST_HU:.0f} HU lesion: "
        "the same filter, two observers"
    )

    cv = lambda a: float(np.std(a) / np.mean(a))  # noqa: E731
    print(f"\nover the kernels whose measured band is intact ({intact.sum()} of {len(rows)}):")
    print(
        f"  noise sd        {noise[intact].min():.1f} .. {noise[intact].max():.1f} HU "
        f"({noise[intact].max() / noise[intact].min():.1f}x)"
    )
    print(f"  NEQ integral    CV = {cv(neq_int[intact]) * 100:.1f} %")
    print(
        f"  d'_ideal        CV = {cv(d_i[intact]) * 100:.1f} %   "
        f"({d_i[intact].min():.3f} .. {d_i[intact].max():.3f})"
    )
    print(
        f"  d'_NPWE         {d_n[intact].min():.3f} .. {d_n[intact].max():.3f}  "
        f"({d_n[intact].max() / d_n[intact].min():.2f}x)"
    )
    fig.suptitle(
        "Real scanner (Siemens Definition Flash), ACR phantom — "
        "reconstruction kernel swept, nothing simulated",
        y=1.02,
    )
    fig.tight_layout()
    out = OUTDIR / "acr_atlas.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
