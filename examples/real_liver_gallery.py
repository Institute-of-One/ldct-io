"""The thing the numbers are about: real liver, real dose reduction, real processing.

Why this figure exists
----------------------
Every other figure in this study is a scatter plot or a synthetic phantom. A reader is asked to
accept that the task measurements describe real low-dose CT without ever being shown one. This
draws the actual images the held-out measurements were made on: one site in one case the networks
never saw, at the routine dose and at the simulated quarter dose, processed by each arm, with the
inserted lesion present in the top row and absent in the bottom row.

What it is meant to show, and it is not flattering to the methods
-----------------------------------------------------------------
The panels are ordered by task detectability, worst on the right. The processed images look
progressively cleaner from left to right while detecting progressively worse, which is the whole
result of the paper in one row. The numbers printed under each panel are the held-out `d'` and
PSNR from ``results/liver_cnn_*.json``: they are read from the recorded runs, not recomputed here,
so the figure and the table cannot disagree.

Every panel shares one display window, chosen once from the reference image, because a denoiser
compared at its own window is being flattered by the display rather than measured.

Data
----
LDCT-and-Projection-data (The Cancer Imaging Archive), CC BY 4.0, DOI 10.7937/9npb-2637. The
figure is a derived illustration of publicly licensed images and carries that attribution in its
caption; no image data is redistributed by this repository.

    python examples/real_liver_gallery.py --case L058
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from denoiq_core.cnn import denoise_stack, load_checkpoint  # noqa: E402
from denoiq_core.denoisers import denoise  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

import sys  # noqa: E402

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
)

from ldct_io import homogeneous_sites, make_paired_trials, noise_only, read_image_series  # noqa: E402

#: Abdominal window for the context slice, in HU: the one a whole abdomen is read at.
WINDOW = (-40.0 - 200.0, -40.0 + 200.0)

#: Width of the window used for the region panels. Its centre is the mean of the lesion-free
#: reference for this site, so the tissue actually fills the greyscale instead of sitting at one
#: end of an abdominal window. The centre and width are the same for every panel in the figure:
#: a denoiser shown at its own window is being flattered by the display rather than measured.
PANEL_WINDOW_WIDTH = 220.0

#: Which arms to draw, as (slug in liver_cnn results, label, how to process).
#: Ordered by held-out detectability, so the eye runs from best task to worst.
ARMS = (
    ("none", "unprocessed", None),
    ("tv 1x noise", "TV", "tv"),
    ("CNN small (21k)", "CNN 21k", "small"),
    ("CNN large (1850k)", "CNN 1.85M", "large"),
    ("GAN large (1850k)", "GAN 1.85M", "large_gan"),
)


def _scores() -> dict[str, dict[str, float]]:
    """Held-out `d'` and PSNR per arm, read from the recorded runs."""
    out: dict[str, dict[str, float]] = {}
    for preset in ("small", "large", "small_gan", "large_gan"):
        path = OUTDIR / f"liver_cnn_{preset}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            out.setdefault(row["label"], {"d_prime": row["d_prime"], "psnr": row["psnr"]})
    return out


def _process(stack: np.ndarray, how: str | None, spacing: float) -> np.ndarray:
    if how is None:
        return stack
    if how == "tv":
        return denoise(stack, method="tv", weight=NOISE_HU)
    model, _ = load_checkpoint(OUTDIR / f"liver_cnn_{how}.pt")
    return denoise_stack(stack, model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default=TEST_CASES[0], help="a held-out case")
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="which paired trial to draw; the default picks the one whose anatomical structure "
        "is closest to the median of the admitted sites, so the panel is representative by a "
        "stated rule rather than by eye",
    )
    parser.add_argument("--out", type=Path, default=OUTDIR / "real_liver_gallery.png")
    args = parser.parse_args(argv)

    if args.case not in TEST_CASES:
        raise SystemExit(
            f"{args.case} is not one of the held-out cases {TEST_CASES}; drawing a training "
            "case would show the networks their own training set"
        )

    full = read_image_series(ROOT / args.case / "full_dose_images")
    low = read_image_series(ROOT / args.case / "low_dose_images")
    difference, _ = noise_only(full, low, dose_fraction=DOSE)
    mid = len(full) // 2
    window = range(max(0, mid - 25), min(len(full), mid + 25))
    sites = homogeneous_sites(
        full.volume, full.spacing, ROI, hu_range=(0.0, 160.0), max_sd=60.0, slices=window
    )
    common = dict(
        size=ROI,
        diameter_mm=LESION_MM,
        contrast_hu=LESION_HU,
        edge_sigma_mm=LESION_EDGE_SIGMA_MM,
        n_trials=min(len(sites), 250),
        seed=SEED,
    )
    trials = make_paired_trials(full.volume, difference, full.spacing, sites, **common)
    clean = make_paired_trials(
        full.volume, np.zeros_like(difference), full.spacing, sites, **common
    )

    # Which site to draw. Choosing by eye would make the panel an argument about the figure
    # rather than about the data, so it is chosen by a rule: the admitted site whose anatomical
    # structure -- the standard deviation after a blur that removes the quantum noise -- is
    # nearest the median of all admitted sites. Neither the cleanest nor the worst.
    half = ROI // 2
    structure = np.array(
        [
            float(
                gaussian_filter(
                    full.volume[s, r - half : r + half, c - half : c + half].astype(np.float64),
                    2.0,
                    mode="nearest",
                ).std()
            )
            for s, r, c in trials.locations
        ]
    )
    median_structure = float(np.median(structure))
    i = int(np.argmin(np.abs(structure - median_structure))) if args.trial is None else int(args.trial)
    present = trials.present[i : i + 1]
    absent = trials.absent[i : i + 1]
    reference = clean.present[i]
    slice_index, row, col = trials.locations[i]
    spacing = trials.spacing
    scores = _scores()

    centre = float(reference.mean())
    panel = (centre - PANEL_WINDOW_WIDTH / 2.0, centre + PANEL_WINDOW_WIDTH / 2.0)

    n = len(ARMS) + 1
    fig = plt.figure(figsize=(2.05 * n, 5.0))
    grid = fig.add_gridspec(2, n, hspace=0.06, wspace=0.04)

    # The context panel: the whole slice, with the region the measurement was made in.
    context = fig.add_subplot(grid[:, 0])
    context.imshow(full.volume[slice_index], cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1])
    context.add_patch(
        plt.Rectangle(
            (col - half, row - half), ROI, ROI, fill=False, edgecolor="C1", lw=1.4
        )
    )
    context.set_title(
        f"{args.case}, routine dose\nslice {slice_index}", fontsize=8, linespacing=1.3
    )
    context.set_xticks([])
    context.set_yticks([])

    for k, (label, name, how) in enumerate(ARMS, start=1):
        shown_present = _process(present, how, spacing)[0]
        shown_absent = _process(absent, how, spacing)[0]
        for r, image in ((0, shown_present), (1, shown_absent)):
            ax = fig.add_subplot(grid[r, k])
            ax.imshow(image, cmap="gray", vmin=panel[0], vmax=panel[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.add_patch(
                    plt.Circle(
                        (ROI / 2 - 0.5, ROI / 2 - 0.5),
                        0.5 * LESION_MM / spacing,
                        fill=False,
                        edgecolor="C1",
                        lw=1.0,
                        alpha=0.9,
                    )
                )
                row_scores = scores.get(label, {})
                caption = name
                if row_scores:
                    caption += (
                        f"\n$d'$ {row_scores['d_prime']:.2f}   {row_scores['psnr']:.1f} dB"
                    )
                ax.set_title(caption, fontsize=8, linespacing=1.3)

    fig.text(0.013, 0.72, "lesion\npresent", fontsize=8, ha="left", va="center")
    fig.text(0.013, 0.28, "lesion\nabsent", fontsize=8, ha="left", va="center")
    fig.suptitle(
        "One held-out site at a quarter of the routine dose: the image improves from left to "
        "right, the task does not",
        fontsize=9,
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.01)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    print(f"  case {args.case}, slice {slice_index}, site ({row}, {col}), trial {i}")
    print(f"  reference mean {reference.mean():.1f} HU, window {WINDOW[0]:.0f} to {WINDOW[1]:.0f}")
    for label, name, _ in ARMS:
        s = scores.get(label)
        if s:
            print(f"  {name:12s} d' {s['d_prime']:.3f}   PSNR {s['psnr']:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
