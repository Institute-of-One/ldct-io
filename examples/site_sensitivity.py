"""Does the held-out ranking survive a site rule that actually selects parenchyma?

Why
---
The admission rule used throughout this study keeps regions of interest whose mean lies in a
tissue window and whose standard deviation is below 60 HU. Drawing the admitted sites showed
what that lets through: bowel, mesentery, vessel and body-wall structure, not liver parenchyma.
Across the held-out cases the anatomical structure inside an admitted site — the standard
deviation of the patch after a mild blur, which is anatomy rather than noise — has a median of
about 20 HU against an inserted lesion 25 HU deep, and exceeds the lesion depth in roughly two
sites in five.

The paired design means this does not bias `d'`: the background is common to the two members of
a pair and cancels, and the noise power spectrum is estimated from paired differences. But the
denoisers are non-linear, so what they do to an image depends on what is in it, and a ranking
measured on structured bowel need not hold on parenchyma. That is the question here, and it is
answered by measurement rather than by argument.

What it does
------------
Re-runs the held-out evaluation under two admission rules, using the networks already trained
and saved, so nothing is retrained and the only thing that differs is where the lesions were put.
Reports both rankings and the rank correlation between them.

    python examples/site_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from denoiq_core.cnn import denoise_stack, load_checkpoint
from denoiq_core.denoisers import denoise
from denoiq_core.evaluate import fidelity
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from taskiq_core import ideal_linear, nps_2d

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

from ldct_io import homogeneous_sites, make_paired_trials, noise_only, read_image_series  # noqa: E402

#: The two admission rules. ``structure_sd`` is applied after the library's own criteria: it is
#: the standard deviation of the ROI after a 2-pixel blur, which removes the quantum noise and
#: leaves the anatomy. Requiring it below 10 HU asks that the structure in the background be
#: well under half the depth of the lesion being looked for, which is what "homogeneous" has to
#: mean if the word is to carry the weight the manuscript puts on it.
#: Only one thing differs between the two rules. A first attempt also narrowed the HU window to
#: 30-90 and the spread cap to 25 HU; that left nineteen sites in one case, and the resulting
#: `d'` of 0.69 against a ceiling of 13.4 measured a prewhitening template estimated from about
#: fifteen trials rather than the task. Scanning the criteria separately showed the structure
#: filter is not what starves the design -- at 10 HU it keeps 294, 229, 282 and 24 sites in the
#: four held-out cases -- so it is applied alone, on the published window, and the comparison is
#: of one variable.
RULES: dict[str, dict[str, Any]] = {
    "as published": dict(hu_range=(0.0, 160.0), max_sd=60.0, max_gradient=40.0, structure_sd=None),
    "low structure": dict(hu_range=(0.0, 160.0), max_sd=60.0, max_gradient=40.0, structure_sd=10.0),
}

NETWORKS = (("small", "CNN 21k"), ("small_gan", "GAN 21k"), ("large", "CNN 1.85M"), ("large_gan", "GAN 1.85M"))


def structure_of(patch: np.ndarray) -> float:
    """Anatomy inside the ROI: what survives a blur that removes the quantum noise."""
    return float(gaussian_filter(patch.astype(np.float64), 2.0, mode="nearest").std())


def gather(
    rule: dict[str, Any], *, max_per_case: int = 250, quota: dict[str, int] | None = None
) -> dict[str, Any]:
    """Sites and trials under one rule.

    ``quota`` caps each case at a given number of pairs. It exists because the strict rule
    admits far fewer sites in one case than in the others, and that case happens to be the
    quietest, so comparing the two rules on their own natural counts would compare a change of
    case mix as much as a change of site. With the quota both rules carry the same pairs per
    case and only the sites differ.
    """
    present, absent, reference, background, patches, used, structure = [], [], [], [], [], {}, []
    signal = spacing = None
    for case in TEST_CASES:
        full = read_image_series(ROOT / case / "full_dose_images")
        low = read_image_series(ROOT / case / "low_dose_images")
        difference, _ = noise_only(full, low, dose_fraction=DOSE)
        mid = len(full) // 2
        window = range(max(0, mid - 25), min(len(full), mid + 25))
        try:
            sites = homogeneous_sites(
                full.volume,
                full.spacing,
                ROI,
                hu_range=rule["hu_range"],
                max_sd=rule["max_sd"],
                max_gradient=rule["max_gradient"],
                slices=window,
            )
        except ValueError:
            print(f"  {case}: no site met the rule", flush=True)
            continue
        half = ROI // 2
        if rule["structure_sd"] is not None:
            keep = [
                k
                for k, (s, r, c) in enumerate(sites)
                if structure_of(full.volume[s, r - half : r + half, c - half : c + half])
                <= rule["structure_sd"]
            ]
            sites = sites[keep]
        if len(sites) < 16:
            print(f"  {case}: only {len(sites)} sites, dropped", flush=True)
            continue
        n_take = min(len(sites), max_per_case)
        if quota is not None:
            if case not in quota:
                print(f"  {case}: not in the quota, dropped", flush=True)
                continue
            n_take = min(n_take, quota[case])
        common = dict(
            size=ROI,
            diameter_mm=LESION_MM,
            contrast_hu=LESION_HU,
            edge_sigma_mm=LESION_EDGE_SIGMA_MM,
            n_trials=n_take,
            seed=SEED,
        )
        trials = make_paired_trials(full.volume, difference, full.spacing, sites, **common)
        clean = make_paired_trials(
            full.volume, np.zeros_like(difference), full.spacing, sites, **common
        )
        present.append(trials.present)
        absent.append(trials.absent)
        reference.append(clean.present)
        background.append(clean.absent)
        patches.append(
            np.stack([difference[s, r - half : r + half, c - half : c + half] for s, r, c in sites])
        )
        structure.extend(
            structure_of(full.volume[s, r - half : r + half, c - half : c + half])
            for s, r, c in sites[:n_take]
        )
        used[case] = n_take
        signal, spacing = trials.signal, trials.spacing
        print(f"  {case}: {len(sites)} sites met the rule, {n_take} used", flush=True)
    if not used:
        raise RuntimeError("no case survived this rule")
    return dict(
        present=np.concatenate(present),
        absent=np.concatenate(absent),
        reference=np.concatenate(reference),
        background=np.concatenate(background),
        patches=np.concatenate(patches).astype(np.float64),
        signal=signal,
        spacing=spacing,
        used=used,
        structure=np.array(structure),
    )


def evaluate(data: dict[str, Any]) -> list[dict[str, Any]]:
    spacing, signal = data["spacing"], data["signal"]
    n = data["present"].shape[0]
    nps = nps_2d(data["patches"], spacing, detrend="mean")
    ceiling = ideal_linear(signal, invertible(nps.nps), spacing, nps_layout="centered").d_prime

    methods: list[tuple[str, Any]] = [
        ("unprocessed", None),
        ("gaussian 0.75 mm", lambda a: denoise(a, method="gaussian", sigma=0.75 / spacing)),
        ("gaussian 1.00 mm", lambda a: denoise(a, method="gaussian", sigma=1.0 / spacing)),
        ("tv 1x noise", lambda a: denoise(a, method="tv", weight=NOISE_HU)),
        ("nlm 0.8x noise", lambda a: denoise(a, method="nlm", h=0.8 * NOISE_HU)),
    ]
    for preset, label in NETWORKS:
        model, _ = load_checkpoint(OUTDIR / f"liver_cnn_{preset}.pt")
        methods.append((label, lambda a, m=model: denoise_stack(a, m)))

    step = max(1, n // 300)
    rows = []
    for label, fn in methods:
        t0 = time.time()
        dp = data["present"] if fn is None else fn(data["present"])
        da = data["absent"] if fn is None else fn(data["absent"])
        d = cross_fitted_paired(dp, da, spacing)
        psnr = float(
            np.mean(
                [
                    fidelity(dp[i][None], data["reference"][i], data_range=400.0)["psnr"]
                    for i in range(0, n, step)
                ]
            )
        )
        rows.append(
            dict(label=label, d_prime=float(d), ratio=float(d / ceiling), psnr=psnr, ceiling=float(ceiling))
        )
        print(
            f"    {label:18s} d' {d:6.3f}  {d / ceiling:5.2f}x  PSNR {psnr:6.2f} dB"
            f"   ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTDIR / "site_sensitivity.json")
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {"rules": {}, "held_out_cases": list(TEST_CASES)}
    # The strict rule is gathered first so its per-case counts become the quota for both, which
    # is what makes the comparison one of sites rather than of case mix.
    probe = gather(RULES["low structure"])
    quota = dict(probe["used"])
    print(f"\nquota from the strict rule: {quota}", flush=True)
    payload["quota"] = quota
    for name, rule in RULES.items():
        print(f"\n=== {name} ===  {rule}", flush=True)
        data = gather(rule, quota=quota)
        print(
            f"  {data['present'].shape[0]} pairs from {len(data['used'])} cases; "
            f"structure sd median {np.median(data['structure']):.1f} HU, "
            f"p90 {np.percentile(data['structure'], 90):.1f}",
            flush=True,
        )
        rows = evaluate(data)
        payload["rules"][name] = {
            "rule": {k: v for k, v in rule.items()},
            "cases": data["used"],
            "n_pairs": int(data["present"].shape[0]),
            "ceiling": rows[0]["ceiling"],
            "structure_sd_median": float(np.median(data["structure"])),
            "structure_sd_p90": float(np.percentile(data["structure"], 90)),
            # How often the anatomy inside a site is deeper than the thing being looked for.
            # The lesion is LESION_HU deep, so this is the fraction of backgrounds carrying
            # structure larger than the signal.
            "structure_over_lesion_fraction": float(
                np.mean(data["structure"] > abs(LESION_HU))
            ),
            "lesion_depth_hu": abs(float(LESION_HU)),
            "rows": rows,
        }

    a, b = payload["rules"]["as published"], payload["rules"]["low structure"]
    labels = [r["label"] for r in a["rows"]]
    da = {r["label"]: r["d_prime"] for r in a["rows"]}
    db = {r["label"]: r["d_prime"] for r in b["rows"]}
    rho, p = spearmanr([da[x] for x in labels], [db[x] for x in labels])
    order_a = sorted(labels, key=lambda x: -da[x])
    order_b = sorted(labels, key=lambda x: -db[x])
    payload["comparison"] = {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "ranking_as_published": order_a,
        "ranking_low_structure": order_b,
        "ranking_identical": order_a == order_b,
        "ceiling_ratio": b["ceiling"] / a["ceiling"],
    }

    print("\n\nranking by d', as published -> parenchyma")
    for i, (x, y) in enumerate(zip(order_a, order_b), start=1):
        mark = "  " if x == y else "<-"
        print(f"  {i}. {x:18s} {mark} {y}")
    print(f"\nSpearman between the two d' vectors: rho = {rho:+.3f} (p = {p:.3f})")
    print(f"ranking identical: {order_a == order_b}")
    print(f"ceiling {a['ceiling']:.3f} -> {b['ceiling']:.3f}")

    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
