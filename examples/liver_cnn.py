"""A network trained on this data, judged by the same ceiling as everything else.

The comparison a reviewer will ask for. A residual CNN is trained on real quarter-dose /
full-dose pairs from eight liver cases and evaluated on four it never saw, beside the
classical denoisers, against the closed-form ceiling of ``liver_ceiling.py``.

What it is, and is not
----------------------
This is ``denoiq_core``'s residual CNN: six layers, 24 channels, 21 385 parameters, trained
here for 16 epochs on 6 400 patches on a CPU. It is *not* RED-CNN, which has roughly eighty
times as many parameters and is trained for far longer. Its validation loss had flattened by
the sixth epoch, so what limits it is capacity rather than training, and a larger network
would certainly reach a higher PSNR. Whether it would also reach a higher detectability is
the question this file is built to ask, and the answer here is only about this network.

Two things are held fixed so the comparison means something:

* **The split is by case, not by patch.** Slices from one patient are not independent, and a
  network tested on a different slice of a liver it trained on is being tested on its own
  training set with extra steps.
* **The normalisation is the one inference uses.** ``denoiq_core.cnn.denoise_stack``
  normalises every image by its own mean and its own estimated noise level, so training pairs
  are built the same way. A network trained in absolute HU and deployed through a normalising
  wrapper is a different network from the one that was trained.

The network never sees a lesion. Its training targets are full-dose reconstructions of
ordinary anatomy, which is what a denoiser is actually given; the lesion exists only in the
evaluation, and only so that the task has a ground truth.

The result
----------
On 1000 held-out pairs from four unseen cases, against a ceiling of 8.09:

    unprocessed          d' 6.03   0.75x   PSNR 23.81 dB
    tv 1x noise          d' 6.09   0.75x   PSNR 28.07 dB
    nlm 0.8x noise       d' 5.50   0.68x   PSNR 27.50 dB
    gaussian 0.75 mm     d' 5.10   0.63x   PSNR 28.04 dB
    CNN                  d' 5.29   0.65x   PSNR 28.71 dB

The network takes the best PSNR of the six, +4.90 dB over the unprocessed input, and gives
back 12 % of the detectability. Total variation, at a *lower* PSNR, gives back none of it.
Ranking these methods by the fidelity metric the denoising literature reports puts them in
close to the opposite order from ranking them by the task, and nothing exceeded the ceiling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from denoiq_core.cnn import (
    DEFAULT_CNN,
    build_cnn,
    denoise_stack,
    estimate_noise_sd,
    save_checkpoint,
)
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
OUTDIR = Path(
    r"C:\Users\YAMAMO~1\AppData\Local\Temp\claude\D--DevGit-DICOM-Viewer"
    r"\4b28a974-1910-45c1-aa13-1fed27b3f70c\scratchpad"
)
CHECKPOINT = OUTDIR / "liver_cnn.pt"

ALL_CASES = sorted(p.name for p in ROOT.glob("L0*"))
TRAIN_CASES, TEST_CASES = ALL_CASES[:8], ALL_CASES[8:]

PATCH, ROI = 64, 48
DOSE = SIMULATED_DOSE_FRACTION["ABDOMEN"]
LESION_MM, LESION_HU, LESION_EDGE_SIGMA_MM = 8.0, -25.0, 0.5
SEED, N_FOLDS = 0, 5
N_PATCHES_PER_CASE, EPOCHS, BATCH, LR = 800, 16, 32, 1e-3
NOISE_HU = 25.0


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


def training_pairs(cases: list[str], *, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Normalised (low-dose, full-dose) patches, exactly as inference will normalise them."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for case in cases:
        t_case = time.time()
        full = read_image_series(ROOT / case / "full_dose_images")
        low = read_image_series(ROOT / case / "low_dose_images")
        if not full.matches(low):
            continue
        n_slices, ny, nx = full.volume.shape
        taken = 0
        attempts = 0
        while taken < N_PATCHES_PER_CASE and attempts < 40 * N_PATCHES_PER_CASE:
            attempts += 1
            s = int(rng.integers(0, n_slices))
            r = int(rng.integers(0, ny - PATCH))
            c = int(rng.integers(0, nx - PATCH))
            target = full.volume[s, r : r + PATCH, c : c + PATCH].astype(np.float64)
            if target.mean() < -300.0:  # mostly air: nothing to learn, and it dominates by area
                continue
            noisy = low.volume[s, r : r + PATCH, c : c + PATCH].astype(np.float64)
            offset = float(noisy.mean())
            scale = float(estimate_noise_sd(noisy))
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            xs.append((noisy - offset) / scale)
            ys.append((target - offset) / scale)
            taken += 1
        print(f"  {case}: {taken} patches ({time.time() - t_case:.0f}s)", flush=True)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def train_network(x: np.ndarray, y: np.ndarray) -> Any:
    torch.manual_seed(SEED)
    model = build_cnn(DEFAULT_CNN, seed=SEED)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    n = x.shape[0]
    cut = int(0.85 * n)
    xt = torch.from_numpy(x[:cut]).unsqueeze(1)
    yt = torch.from_numpy(y[:cut]).unsqueeze(1)
    xv = torch.from_numpy(x[cut:]).unsqueeze(1)
    yv = torch.from_numpy(y[cut:]).unsqueeze(1)
    print(f"\ntraining on {cut} patches, validating on {n - cut}")

    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        order = torch.randperm(cut, generator=generator)
        total = 0.0
        for start in range(0, cut, BATCH):
            index = order[start : start + BATCH]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[index]), yt[index])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            validation = float(loss_fn(model(xv), yv))
        print(
            f"  epoch {epoch + 1:2d}/{EPOCHS}  train {total / cut:.5f}  "
            f"val {validation:.5f}   ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return model


def gather_test() -> dict[str, Any]:
    present, absent, reference, patches, used = [], [], [], [], {}
    signal = spacing = None
    for case in TEST_CASES:
        full = read_image_series(ROOT / case / "full_dose_images")
        low = read_image_series(ROOT / case / "low_dose_images")
        if not full.matches(low):
            continue
        noise, _ = noise_only(full, low, dose_fraction=DOSE)
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
    return dict(
        present=np.concatenate(present),
        absent=np.concatenate(absent),
        reference=np.concatenate(reference),
        patches=np.concatenate(patches).astype(np.float64),
        signal=signal,
        spacing=spacing,
        used=used,
    )


def main() -> int:
    print(f"train on {TRAIN_CASES}\ntest on  {TEST_CASES}\n")
    x, y = training_pairs(TRAIN_CASES)
    print(f"\n{x.shape[0]} training patches of {PATCH}x{PATCH}, normalised as at inference")
    model = train_network(x, y)
    digest = save_checkpoint(
        model,
        CHECKPOINT,
        extra={"train_cases": TRAIN_CASES, "epochs": EPOCHS, "seed": SEED, "patch": PATCH},
    )
    print(f"checkpoint sha256 {digest}")

    data = gather_test()
    spacing, signal = data["spacing"], data["signal"]
    noise_nps = nps_2d(data["patches"], spacing, detrend="mean")
    ceiling = ideal_linear(
        signal, invertible(noise_nps.nps), spacing, nps_layout="centered"
    ).d_prime
    n = data["present"].shape[0]
    print(
        f"\nheld-out test: {n} pairs from {len(data['used'])} cases, "
        f"noise sd {np.sqrt(noise_nps.integral):.1f} HU, ceiling d' = {ceiling:.3f}\n"
    )

    methods: list[tuple[str, Any]] = [
        ("none", None),
        ("gaussian 0.75 mm", lambda a: denoise(a, method="gaussian", sigma=0.75 / spacing)),
        ("gaussian 1.00 mm", lambda a: denoise(a, method="gaussian", sigma=1.0 / spacing)),
        ("tv 1x noise", lambda a: denoise(a, method="tv", weight=NOISE_HU)),
        ("nlm 0.8x noise", lambda a: denoise(a, method="nlm", h=0.8 * NOISE_HU)),
        ("CNN (trained here)", lambda a: denoise_stack(a, model)),
    ]

    rows = []
    for label, fn in methods:
        t0 = time.time()
        dp = data["present"] if fn is None else fn(data["present"])
        da = data["absent"] if fn is None else fn(data["absent"])
        d = cross_fitted_paired(dp, da, spacing)
        step = max(1, n // 300)
        psnr = float(
            np.mean(
                [
                    fidelity(dp[i][None], data["reference"][i], data_range=400.0)["psnr"]
                    for i in range(0, n, step)
                ]
            )
        )
        rows.append(dict(label=label, d_prime=float(d), ratio=float(d / ceiling), psnr=psnr))
        print(
            f"  {label:20s} d' {d:6.3f}   {d / ceiling:5.2f}x ceiling   PSNR {psnr:6.2f} dB"
            f"   ({time.time() - t0:.0f}s)",
            flush=True,
        )

    above = [r for r in rows if r["ratio"] > 1.02]
    print(
        f"\nabove the ceiling: {len(above)} of {len(rows)}"
        + (f" -> {[r['label'] for r in above]}" if above else "")
    )
    raw = rows[0]
    best_psnr = max(rows, key=lambda r: r["psnr"])
    print(
        f"best PSNR: {best_psnr['label']} at {best_psnr['psnr']:.2f} dB "
        f"({best_psnr['psnr'] - raw['psnr']:+.2f} dB), "
        f"d' {best_psnr['d_prime'] / raw['d_prime']:.2f}x the unprocessed input"
    )

    (OUTDIR / "liver_cnn.json").write_text(
        json.dumps(
            {
                "train": TRAIN_CASES,
                "test": data["used"],
                "sha256": digest,
                "ceiling": ceiling,
                "rows": rows,
            },
            indent=2,
        )
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colours = ["0.5"] * (len(rows) - 1) + ["C2"]
    ax.scatter([r["psnr"] for r in rows], [r["d_prime"] for r in rows], c=colours, s=70, zorder=3)
    for r in rows:
        ax.annotate(
            r["label"],
            (r["psnr"], r["d_prime"]),
            fontsize=8,
            textcoords="offset points",
            xytext=(6, 4),
        )
    ax.axhline(ceiling, color="C3", lw=2, label=f"closed-form ceiling ({ceiling:.2f})")
    ax.set_xlabel("PSNR against the full-dose reconstruction [dB]")
    ax.set_ylabel(r"held-out $d'$, background known exactly")
    ax.set_ylim(0, ceiling * 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{n} held-out pairs, {len(data['used'])} liver cases the network never saw")
    fig.tight_layout()
    out = OUTDIR / "liver_cnn.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
