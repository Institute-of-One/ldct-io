"""A network trained on this data, judged by the same ceiling as everything else.

The comparison a reviewer will ask for. A residual CNN is trained on real quarter-dose /
full-dose pairs from eight liver cases and evaluated on four it never saw, beside the
classical denoisers, against the closed-form ceiling of ``liver_ceiling.py``.

Run it at either capacity::

    python examples/liver_cnn.py --preset small   # 21 k parameters
    python examples/liver_cnn.py --preset large   # 1.85 M, RED-CNN's scale

Two capacities, because "the network did worse on the task" has to be told apart from "the
network was too small".

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
1000 held-out pairs from four unseen cases, ceiling d' = 8.087:

    method              d'      of ceiling   PSNR
    tv 1x noise         6.09      0.75x      28.07 dB
    unprocessed         6.03      0.75x      23.81 dB
    nlm 0.8x noise      5.50      0.68x      27.50 dB
    CNN 21 k            5.31      0.66x      28.74 dB
    gaussian 0.75 mm    5.10      0.63x      28.04 dB
    CNN 1.85 M          4.96      0.61x      28.77 dB   <- best PSNR of all seven
    gaussian 1.00 mm    4.83      0.60x      27.85 dB

    above the ceiling: 0 of 7
    Spearman(PSNR, d') across the seven methods: -0.29

Eighty-seven times the parameters, three and a half times the training data, a genuinely
better validation loss (5.00 against 5.67) -- and 0.03 dB more PSNR for 7 % less
detectability than the small network. Raising capacity improved the objective the network
was trained on and made the task worse. The ranking does not flip, so the earlier result was
not an artefact of a small network.

Two honest notes. The large network's validation loss bottomed at epoch 20 and drifted up
slightly to epoch 60, so the saved model is not quite the best one; best-epoch checkpointing
would recover about 0.06 of validation loss, which is far too little to move d'. And +0.03 dB
for 87x the capacity is itself worth noticing: the training target is a *full-dose
reconstruction*, which has noise of its own that no network can predict, so mean-squared
error against it saturates well before the image does.
"""

from __future__ import annotations

import argparse
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
    CNNConfig,
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
ALL_CASES = sorted(p.name for p in ROOT.glob("L0*"))
TRAIN_CASES, TEST_CASES = ALL_CASES[:8], ALL_CASES[8:]

PATCH, ROI = 64, 48
DOSE = SIMULATED_DOSE_FRACTION["ABDOMEN"]
LESION_MM, LESION_HU, LESION_EDGE_SIGMA_MM = 8.0, -25.0, 0.5
SEED, N_FOLDS = 0, 5
NOISE_HU = 25.0

#: Two capacities, so that "the network did worse on the task" can be told apart from "the
#: network was too small". The large one is RED-CNN's scale: ten layers, 96 channels, 5x5
#: kernels, about 1.85 M parameters against the small one's 21 k.
PRESETS: dict[str, dict[str, Any]] = {
    "small": dict(
        cnn=CNNConfig(depth=6, width=24, kernel_size=3, residual=True),
        patches=800,
        epochs=16,
        batch=32,
        lr=1e-3,
    ),
    "large": dict(
        cnn=CNNConfig(depth=10, width=96, kernel_size=5, residual=True),
        patches=3000,
        epochs=60,
        batch=64,
        lr=1e-3,
    ),
}


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


def training_pairs(
    cases: list[str], n_per_case: int, *, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
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
        while taken < n_per_case and attempts < 40 * n_per_case:
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


def train_network(
    x: np.ndarray, y: np.ndarray, preset: dict[str, Any], device: str
) -> tuple[Any, int, float]:
    """Train, on the GPU if there is one, and hand the model back on the CPU.

    Inference goes through ``denoiq_core.cnn.denoise_stack``, which feeds CPU tensors, so the
    model comes home before it is used. One inference path for both capacities, and for
    anyone re-running this without a GPU.
    """
    torch.manual_seed(SEED)
    model = build_cnn(preset["cnn"], seed=SEED).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.Adam(model.parameters(), lr=preset["lr"])
    loss_fn = torch.nn.MSELoss()

    n = x.shape[0]
    cut = int(0.85 * n)
    xt = torch.from_numpy(x[:cut]).unsqueeze(1).to(device)
    yt = torch.from_numpy(y[:cut]).unsqueeze(1).to(device)
    xv = torch.from_numpy(x[cut:]).unsqueeze(1).to(device)
    yv = torch.from_numpy(y[cut:]).unsqueeze(1).to(device)
    batch, epochs = preset["batch"], preset["epochs"]
    print(
        f"\n{n_parameters} parameters on {device}; training on {cut} patches, "
        f"validating on {n - cut}",
        flush=True,
    )

    generator = torch.Generator().manual_seed(SEED)
    best = float("inf")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order = torch.randperm(cut, generator=generator).to(device)
        total = 0.0
        for start in range(0, cut, batch):
            index = order[start : start + batch]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[index]), yt[index])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            validation = float(
                sum(
                    float(loss_fn(model(xv[i : i + batch]), yv[i : i + batch]))
                    * xv[i : i + batch].shape[0]
                    for i in range(0, xv.shape[0], batch)
                )
                / xv.shape[0]
            )
        best = min(best, validation)
        if epoch < 3 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch + 1:3d}/{epochs}  train {total / cut:.5f}  "
                f"val {validation:.5f}   ({time.time() - t0:.1f}s)",
                flush=True,
            )
    print(f"  best validation loss {best:.5f}", flush=True)
    return model.cpu(), n_parameters, best


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    print(f"preset {args.preset}: {preset['cnn']}")
    print(f"train on {TRAIN_CASES}\ntest on  {TEST_CASES}\n", flush=True)
    x, y = training_pairs(TRAIN_CASES, preset["patches"])
    print(f"\n{x.shape[0]} training patches of {PATCH}x{PATCH}, normalised as at inference")
    model, n_parameters, best_val = train_network(x, y, preset, device)
    checkpoint = OUTDIR / f"liver_cnn_{args.preset}.pt"
    digest = save_checkpoint(
        model,
        checkpoint,
        extra={
            "train_cases": TRAIN_CASES,
            "preset": args.preset,
            "epochs": preset["epochs"],
            "seed": SEED,
            "patch": PATCH,
        },
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
        (f"CNN {args.preset} ({n_parameters / 1000:.0f}k)", lambda a: denoise_stack(a, model)),
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

    (OUTDIR / f"liver_cnn_{args.preset}.json").write_text(
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
    out = OUTDIR / f"liver_cnn_{args.preset}.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
