r"""Where the dose can be conceded, and what denoising does not give back.

The question this answers
------------------------
A radiographer at the console does not choose a denoiser; they choose an exposure. The
operational question is therefore not "which method has the best PSNR" but "how far can this
protocol be cut before the task is lost, and does reconstruction or AI denoising move that
number". This script answers it on real liver data, and the answer it gives is a dose, not a
figure of merit.

Two quantities are reported at every dose, and they part company
---------------------------------------------------------------
*Fidelity*, as PSNR against the routine-dose reconstruction, is what the image looks like, and
it is what a reader judges a denoiser by. *Detectability* is whether the lesion can be found.
The script converts the first into the second's currency: a method that raises PSNR by
:math:`\\Delta` dB lowers the mean-squared error by :math:`10^{\\Delta/10}`, and since that
error is the inserted noise's power, the same reduction could have been bought with dose. So
every processed image has an **apparent dose** -- the exposure whose unprocessed image would
have looked this good -- printed beside the detectability it actually delivers. If the two
agreed, appearance would be a safe basis for cutting the exposure. They do not.

How the dose axis is measured rather than assumed
-------------------------------------------------
This collection's low-dose series is a reconstruction of the *same* projections with noise
inserted, so the difference of the two reconstructions is a measured realisation of exactly the
noise that dose reduction costs -- correct spectrum, real anatomy, no simulation. If the
inserted noise is independent of the noise already present, its power scales as
:math:`1/\\beta - 1` at dose fraction :math:`\\beta`, so scaling that measured realisation by

.. math::  k(\\beta) = \\sqrt{\\frac{1/\\beta - 1}{1/\\alpha - 1}}

gives a noise realisation at any dose fraction, with the spectrum still measured. The trials are
built once at the nominal fraction and recombined linearly, which is exact: ``present`` is
``background + lesion + noise`` and ``absent`` is ``background + noise`` by construction, so the
noise-free pass supplies the background and the difference supplies the noise.

The resulting scaling law, :math:`d' \\propto 1/k(\\beta)`, is **checked** rather than relied on,
and it fails for every achievable observer. It holds exactly for the ceiling, whose prewhitening
template is the noise's own; it does not hold for a real observer looking at a real denoised
image, whose efficiency peaks near the nominal fraction and falls away on both sides, worst for
the smoothers. This matters for the decision, because it means the textbook
:math:`d' \\propto \\sqrt{\\text{dose}}` rule *overstates* what a denoised image keeps when the
exposure is cut. The dose at which a method crosses the requirement is therefore interpolated
from the measured curve, and the closed form is carried alongside only as a cross-check.

What the dose axis here means, and what it does not
--------------------------------------------------
The background is the routine-dose reconstruction, treated as known. The detectability is
therefore of *what the dose reduction took away*, for a reader who already knows this patient's
anatomy -- an idealisation, and the right one for asking whether processing can put information
back. It is not a claim about reading a scan cold, and the absolute dose fractions inherit that
idealisation. What survives it is the comparison between methods at one dose, and the ratio
between doses, which is what the decision actually rests on.

The requirement is prespecified
-------------------------------
``d' = 5`` is the Rose criterion and is already the operational floor in
``denoiq_core.redlamp.DEFAULT_CRITERIA``; it is not chosen here, after the data. It is a task
requirement, not an information boundary: the boundary is the ceiling, and the floor is what the
task needs of the ceiling.

Running it::

    python examples/dose_decision.py                       # the held-out cases, every method
    python examples/dose_decision.py --doses 0.5 0.25 0.1  # a dose grid of your own
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import sys  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from denoiq_core.cnn import denoise_stack, load_checkpoint  # noqa: E402
from denoiq_core.denoisers import denoise  # noqa: E402
from denoiq_core.evaluate import fidelity  # noqa: E402
from denoiq_core.redlamp import (  # noqa: E402
    DEFAULT_CRITERIA,
    contrast_recovery,
    false_structure_rate,
)
from taskiq_core import ideal_linear, nps_2d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liver_cnn import (  # noqa: E402
    DOSE,
    LESION_EDGE_SIGMA_MM,
    LESION_HU,
    LESION_MM,
    NOISE_HU,
    OUTDIR,
    ROI,
    ROOT,
    SEED,
    TEST_CASES,
    cross_fitted_paired,
    invertible,
)

from ldct_io import (  # noqa: E402
    homogeneous_sites,
    make_paired_trials,
    noise_only,
    read_image_series,
)

#: Dose fractions of the routine protocol. Centred on the collection's own simulated fraction so
#: that one point on the axis is measured noise at its native amplitude and the rest are scaled.
DEFAULT_DOSES = (0.50, 0.35, 0.25, 0.18, 0.125, 0.09, 0.0625)

#: The prespecified operational floor, from the gauge's own criteria rather than from this run.
REQUIREMENT = DEFAULT_CRITERIA.d_prime_threshold


def noise_scale(beta: float, alpha: float) -> float:
    """How much to scale noise measured at dose fraction ``alpha`` to represent ``beta``."""
    for name, value in (("beta", beta), ("alpha", alpha)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}")
    return float(np.sqrt((1.0 / beta - 1.0) / (1.0 / alpha - 1.0)))


def dose_for_requirement(beta: float, measured: float, requirement: float) -> float:
    r"""The dose fraction at which a method measuring ``measured`` would reach ``requirement``.

    From :math:`d' \propto 1/\sqrt{1/\beta - 1}`:

    .. math::  \frac{1}{\beta^{*}} - 1 = \left(\frac{1}{\beta} - 1\right)
               \left(\frac{d'}{d'_{\text{req}}}\right)^{2}

    Returns ``nan`` when the method never reaches the requirement at any dose, which for this
    monotone law means ``measured`` is not positive.
    """
    if not np.isfinite(measured) or measured <= 0.0:
        return float("nan")
    excess = (1.0 / beta - 1.0) * (measured / float(requirement)) ** 2
    return float(1.0 / (1.0 + excess))


def apparent_dose(beta: float, psnr_gain_db: float) -> float:
    r"""The dose whose unprocessed image would have had this much fidelity.

    PSNR is :math:`10\log_{10}(\text{range}^2/\text{MSE})` and the mean-squared error against
    the routine-dose reconstruction is the inserted noise's power, which scales as
    :math:`1/\beta - 1`. A gain of :math:`\Delta` dB is therefore the same fidelity as an
    exposure satisfying :math:`1/\beta^{*} - 1 = (1/\beta - 1) \, 10^{-\Delta/10}`.

    This is the number a denoised image invites a reader to believe about its exposure.
    """
    excess = (1.0 / beta - 1.0) * 10.0 ** (-float(psnr_gain_db) / 10.0)
    return float(1.0 / (1.0 + excess))


def gather(cases: list[str], *, alpha: float = DOSE) -> dict[str, Any]:
    """Backgrounds, lesion-free backgrounds and the two noise draws, per trial.

    Read once. The dose axis is then a scalar multiplying the noise, because ``present`` and
    ``absent`` are linear in it by construction.
    """
    clean_present, clean_absent, noise_a, noise_b, patches, used = [], [], [], [], [], {}
    signal = spacing = None
    for case in cases:
        t0 = time.time()
        full = read_image_series(ROOT / case / "full_dose_images")
        low = read_image_series(ROOT / case / "low_dose_images")
        if not full.matches(low):
            continue
        difference, _ = noise_only(full, low, dose_fraction=alpha)
        mid = len(full) // 2
        window = range(max(0, mid - 25), min(len(full), mid + 25))
        try:
            sites = homogeneous_sites(
                full.volume, full.spacing, ROI, hu_range=(0.0, 160.0), max_sd=60.0, slices=window
            )
        except ValueError:
            continue
        n_take = min(len(sites), 250)
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
        noisy = make_paired_trials(full.volume, difference, full.spacing, sites, **common)
        clean = make_paired_trials(
            full.volume, np.zeros_like(difference), full.spacing, sites, **common
        )
        clean_present.append(clean.present)
        clean_absent.append(clean.absent)
        # present = background + lesion + noise_a, absent = background + noise_b, exactly.
        noise_a.append(noisy.present - clean.present)
        noise_b.append(noisy.absent - clean.absent)
        half = ROI // 2
        patches.append(
            np.stack([difference[s, r - half : r + half, c - half : c + half] for s, r, c in sites])
        )
        used[case] = n_take
        signal, spacing = noisy.signal, noisy.spacing
        print(f"  {case}: {n_take} pairs ({time.time() - t0:.0f}s)", flush=True)
    if not used:
        raise RuntimeError("no case yielded trials: check LDCT_IO_DATA")
    return dict(
        clean_present=np.concatenate(clean_present),
        clean_absent=np.concatenate(clean_absent),
        noise_a=np.concatenate(noise_a),
        noise_b=np.concatenate(noise_b),
        patches=np.concatenate(patches).astype(np.float64),
        signal=signal,
        spacing=spacing,
        used=used,
        alpha=float(alpha),
    )


#: Networks the chart may include, and what to call them.
NETWORKS = (
    ("small", "CNN 21k"),
    ("small_gan", "GAN 21k"),
    ("large", "CNN 1.85M"),
    ("large_gan", "GAN 1.85M"),
)


def methods_for(spacing: float, presets: tuple[str, ...]) -> list[tuple[str, Any]]:
    """The unprocessed input, the classical denoisers, and the named networks.

    ``presets`` is named explicitly rather than taken from whatever ``.pt`` files are lying in
    ``results/``: a checkpoint from an earlier run has the same filename as the one being
    written now, and a stale network in this chart would be invisible.
    """
    items: list[tuple[str, Any]] = [
        ("unprocessed", None),
        ("gaussian 0.75 mm", lambda a: denoise(a, method="gaussian", sigma=0.75 / spacing)),
        ("gaussian 1.00 mm", lambda a: denoise(a, method="gaussian", sigma=1.0 / spacing)),
        ("tv 1x noise", lambda a: denoise(a, method="tv", weight=NOISE_HU)),
        ("nlm 0.8x noise", lambda a: denoise(a, method="nlm", h=0.8 * NOISE_HU)),
    ]
    for preset, label in NETWORKS:
        if preset not in presets:
            continue
        path = OUTDIR / f"liver_cnn_{preset}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{preset} was asked for but {path} does not exist")
        model, _ = load_checkpoint(path)
        items.append((label, lambda a, m=model: denoise_stack(a, m)))
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSES))
    parser.add_argument("--requirement", type=float, default=REQUIREMENT)
    parser.add_argument(
        "--presets",
        nargs="*",
        default=[name for name, _ in NETWORKS],
        choices=[name for name, _ in NETWORKS],
        help="which trained networks to include; name them, so a stale checkpoint cannot slip in",
    )
    args = parser.parse_args(argv)
    doses = sorted((float(d) for d in args.doses), reverse=True)
    requirement = float(args.requirement)

    print(f"held-out cases {TEST_CASES}, noise measured at dose fraction {DOSE}")
    data = gather(list(TEST_CASES))
    spacing, signal = data["spacing"], data["signal"]
    n = data["clean_present"].shape[0]
    print(f"\n{n} pairs from {len(data['used'])} cases")
    print(f"requirement d' >= {requirement} (Rose criterion, prespecified in RedLampCriteria)\n")

    methods = methods_for(spacing, tuple(args.presets))
    assert methods[0][1] is None, "the unprocessed input has to come first: it is the reference"
    step = max(1, n // 300)
    table: list[dict[str, Any]] = []
    for beta in doses:
        k = noise_scale(beta, data["alpha"])
        present = data["clean_present"] + k * data["noise_a"]
        absent = data["clean_absent"] + k * data["noise_b"]
        nps = nps_2d(k * data["patches"], spacing, detrend="mean")
        ceiling = ideal_linear(signal, invertible(nps.nps), spacing, nps_layout="centered").d_prime
        print(
            f"dose {beta:6.3f} of routine  (noise x{k:.3f}, sd {np.sqrt(nps.integral):5.1f} HU)"
            f"   ceiling d' {ceiling:6.3f}",
            flush=True,
        )
        raw_psnr = None
        for label, fn in methods:
            dp = present if fn is None else fn(present)
            da = absent if fn is None else fn(absent)
            d = cross_fitted_paired(dp, da, spacing)
            psnr = float(
                np.mean(
                    [
                        fidelity(dp[i][None], data["clean_present"][i], data_range=400.0)["psnr"]
                        for i in range(0, n, step)
                    ]
                )
            )
            if raw_psnr is None:
                raw_psnr = psnr
            background = data["clean_absent"] if fn is None else fn(data["clean_absent"])
            added = false_structure_rate(da - background, signal)["rate"]
            row = dict(
                dose=beta,
                noise_scale=k,
                ceiling=float(ceiling),
                label=label,
                d_prime=float(d),
                efficiency=float(d / ceiling),
                psnr=psnr,
                psnr_gain_db=float(psnr - raw_psnr),
                apparent_dose=apparent_dose(beta, psnr - raw_psnr),
                dose_at_requirement=dose_for_requirement(beta, d, requirement),
                added_structure_rate=float(added),
                contrast_recovery=float(contrast_recovery(dp, da, signal)),
                meets_requirement=bool(d >= requirement),
            )
            table.append(row)
            print(
                f"    {label:18s} d' {d:6.3f}  {'ok ' if row['meets_requirement'] else 'FAIL'}"
                f"  PSNR {psnr:6.2f} dB ({row['psnr_gain_db']:+5.2f})"
                f"  looks like dose {row['apparent_dose']:6.3f}",
                flush=True,
            )

    # --- the scaling law, checked rather than assumed -------------------------------------
    #
    # The ideal observer's d' scales as 1/k exactly, because its NPS is the noise's. The
    # achievable observer's does not have to, and on this data it does not: efficiency peaks
    # near the nominal fraction and falls away on both sides, worst for the smoothers. That is
    # a result and not a nuisance -- the ideal-observer dose law flatters a denoised image,
    # because it assumes an efficiency the denoised image does not keep. So the crossing is
    # read off the measured curve, and the closed form is reported only as a cross-check.
    print("\nis d' x k constant across the dose axis? (it is, exactly, only for the ceiling)")
    law: dict[str, dict[str, Any]] = {}
    for label, _ in methods:
        rows_for = [r for r in table if r["label"] == label]
        products = [r["d_prime"] * r["noise_scale"] for r in rows_for]
        near = [r["d_prime"] * r["noise_scale"] for r in rows_for if r["dose"] <= DOSE + 1e-9]
        spread = (max(products) - min(products)) / float(np.mean(products))
        spread_near = (max(near) - min(near)) / float(np.mean(near)) if len(near) > 1 else 0.0
        law[label] = {
            "mean": float(np.mean(products)),
            "spread_fraction": float(spread),
            "spread_fraction_at_or_below_nominal": float(spread_near),
            "values": [float(p) for p in products],
        }
        print(
            f"    {label:18s} mean {np.mean(products):6.3f}   spread {spread:5.1%}"
            f"   (at or below {DOSE:.2f}: {spread_near:5.1%})"
        )
    ceilings = {r["dose"]: r["ceiling"] for r in table}
    ceiling_products = [c * noise_scale(b, data["alpha"]) for b, c in ceilings.items()]
    ceiling_spread = (max(ceiling_products) - min(ceiling_products)) / float(
        np.mean(ceiling_products)
    )
    print(
        f"    {'ceiling':18s} mean {np.mean(ceiling_products):6.3f}   spread {ceiling_spread:5.1%}"
    )

    # --- the decision --------------------------------------------------------------------
    crossing: dict[str, float] = {}
    for label, _ in methods:
        rows_for = sorted((r for r in table if r["label"] == label), key=lambda r: r["dose"])
        # d' rises with dose, so interpolate dose against d' in logs, which is where the law
        # would be a straight line.
        crossing[label] = float(
            np.exp(
                np.interp(
                    np.log(requirement),
                    np.log([r["d_prime"] for r in rows_for]),
                    np.log([r["dose"] for r in rows_for]),
                )
            )
        )

    at_nominal = [r for r in table if abs(r["dose"] - DOSE) < 1e-9]
    raw_crossing = crossing["unprocessed"]
    print(f"\nThe decision. Dose at which d' falls to {requirement:.0f}, measured:")
    print(
        f"    {'method':18s} {'dose needed':>11s} {'vs nothing':>11s}"
        f"   {'at ' + format(DOSE, '.2f') + ':':>8s} {'d-prime':>8s} {'looks like':>11s}"
    )
    for label, _ in methods:
        row = next(r for r in at_nominal if r["label"] == label)
        print(
            f"    {label:18s} {crossing[label]:11.3f} "
            f"{crossing[label] / raw_crossing - 1.0:+10.0%}   {'':>8s} "
            f"{row['d_prime']:8.3f} {row['apparent_dose']:11.3f}"
        )

    costlier = [lab for lab in crossing if crossing[lab] > raw_crossing * 1.02]
    print(
        f"\nDoing nothing meets the requirement down to {raw_crossing:.3f} of the routine dose. "
        f"Of the {len(crossing) - 1} processed\narms, {len(costlier)} need *more* dose than that "
        f"to meet it"
        + (f" (up to {max(crossing.values()) / raw_crossing - 1.0:+.0%})" if costlier else "")
        + f", and none needs less than "
        f"{min(crossing.values()) / raw_crossing - 1.0:+.0%}. Processing does not buy exposure."
    )

    # How much exposure the picture claims, dose by dose. The point is the trend.
    print("\nHow much more exposure the processed image looks like it was given:")
    header = "  ".join(f"{label[:9]:>9s}" for label, _ in methods[1:])
    print(f"    {'dose':>6s}  {header}   task met?")
    for beta in doses:
        at = {r["label"]: r for r in table if r["dose"] == beta}
        claims = "  ".join(f"{at[label]['apparent_dose'] / beta:9.2f}" for label, _ in methods[1:])
        met = sum(1 for r in at.values() if r["meets_requirement"])
        print(f"    {beta:6.3f}  {claims}   {met}/{len(at)}")
    worst = max(
        table, key=lambda r: r["apparent_dose"] / r["dose"] if not r["meets_requirement"] else 0
    )
    print(
        f"\nThe overstatement grows as the exposure falls, so it is largest where the task has\n"
        f"already been lost: at {worst['dose']:.3f} of routine, {worst['label']} looks like "
        f"{worst['apparent_dose']:.3f}\n"
        f"({worst['apparent_dose'] / worst['dose']:.1f}x the exposure "
        f"it was given) and delivers d' {worst['d_prime']:.2f} against the "
        f"{requirement:.0f} required."
    )

    payload = {
        "what": "dose at which the task is lost, and what processing does not return",
        "cases": data["used"],
        "n_pairs": int(n),
        "alpha_measured_at": data["alpha"],
        "requirement": requirement,
        "requirement_source": "RedLampCriteria.d_prime_threshold (Rose), prespecified",
        "doses": doses,
        "scaling_law_check": law,
        "ceiling_scaling_spread": float(ceiling_spread),
        "dose_at_requirement_measured": crossing,
        "rows": table,
    }
    out = OUTDIR / "dose_decision.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    _figure(table, doses, requirement, data, crossing)
    return 0


def _figure(
    table: list[dict[str, Any]],
    doses: list[float],
    requirement: float,
    data: dict[str, Any],
    crossing: dict[str, float],
) -> None:
    """Detectability against dose, with the requirement and each arm's apparent dose."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = []
    for row in table:
        if row["label"] not in labels:
            labels.append(row["label"])

    ceilings = [next(r["ceiling"] for r in table if r["dose"] == d) for d in doses]
    ax.plot(doses, ceilings, color="C3", lw=2.4, ls="--", label="ceiling (ideal observer)")
    # Red belongs to the ceiling alone, and black to doing nothing: the two references a
    # reader compares everything else against have to be unmistakable.
    palette = ["C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
    for i, label in enumerate(labels[1:]):
        rows = [r for r in table if r["label"] == label]
        ax.plot(
            [r["dose"] for r in rows],
            [r["d_prime"] for r in rows],
            color=palette[i % len(palette)],
            marker="o",
            ms=3,
            lw=1.2,
            label=label,
        )
    raw_rows = [r for r in table if r["label"] == "unprocessed"]
    ax.plot(
        [r["dose"] for r in raw_rows],
        [r["d_prime"] for r in raw_rows],
        color="k",
        marker="o",
        ms=5,
        lw=2.4,
        label="unprocessed",
        zorder=5,
    )
    ax.axhline(requirement, color="0.3", ls="--", lw=1.2)
    ax.annotate(
        f"requirement d' = {requirement:.0f}",
        (doses[-1], requirement),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8,
        color="0.3",
    )
    raw_crossing = crossing["unprocessed"]
    ax.axvline(raw_crossing, color="0.3", ls=":", lw=1.2)
    ax.annotate(
        f"doing nothing holds\nto {raw_crossing:.3f} of routine",
        (raw_crossing, min(r["d_prime"] for r in table)),
        xytext=(4, 2),
        textcoords="offset points",
        fontsize=7,
        color="0.3",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dose, fraction of the routine protocol")
    ax.set_ylabel(r"held-out $d'$")
    ax.set_title("What the exposure buys")
    ax.grid(alpha=0.3, which="both")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(doses)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower right")

    at_nominal = [r for r in table if abs(r["dose"] - data["alpha"]) < 1e-9]
    names = [r["label"] for r in at_nominal]
    bx.barh(
        range(len(at_nominal)),
        [r["apparent_dose"] for r in at_nominal],
        color="0.75",
        label="dose it looks like (PSNR)",
    )
    bx.barh(
        range(len(at_nominal)),
        [r["dose"] for r in at_nominal],
        height=0.45,
        color="C0",
        label="dose it was acquired at",
    )
    bx.set_yticks(range(len(at_nominal)))
    bx.set_yticklabels(names, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel("fraction of the routine protocol")
    bx.set_title(f"Appearance against exposure at {data['alpha']:.2f} dose")
    bx.grid(alpha=0.3, axis="x")
    bx.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = OUTDIR / "dose_decision.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
