"""Locate the adversarial weight at which the generator starts synthesising texture.

Why a sweep and not a chosen number
-----------------------------------
The adversarial arm exists to supply a denoiser that can fabricate structure, so that the
ceiling argument is tested against something capable of breaking it. If the adversarial weight
is too small the generator is an MSE network under another name and the whole comparison is
between a denoiser and itself; if it is too large the generator stops tracking its input and the
experiment measures a collapse. Either way the result would be an artefact of a number somebody
picked. This script records the sweep that picks it.

What it measures, per weight
----------------------------
``val_mse``
    Fidelity. Training for appearance has to cost fidelity; if it does not, the adversarial term
    is acting as a regulariser and there is no mechanism behind "fidelity rose while the task
    fell".

``texture_ratio``
    The diagnostic that separates the two failure modes from the regime of interest. It is the
    standard deviation of the output's fine detail divided by the same quantity for the
    full-dose target, where fine detail is what a Gaussian of one pixel does not keep. An
    MSE-trained denoiser over-smooths, so the ratio sits well below one. A generator that has
    learned to produce full-dose-looking texture sits near one. A collapsed generator produces
    detail unrelated to the target and can sit anywhere, which is why ``val_mse`` is read
    alongside it and not instead of it.

``registration``
    Pearson correlation between output and target over the patch. It catches the collapse
    ``texture_ratio`` alone would miss: a generator emitting plausible liver texture that has
    nothing to do with *this* liver still has a ratio near one, and its correlation falls.

Running it::

    python examples/adv_weight_sweep.py                    # the default grid
    python examples/adv_weight_sweep.py --weights 0 0.1 1  # a grid of your own

It trains on a subset by design: the question is where the regime boundaries are, not what the
final model is, and the final model is trained by ``liver_cnn.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liver_cnn import OUTDIR, PRESETS, SEED, TRAIN_CASES, training_pairs  # noqa: E402

from denoiq_core.cnn import build_cnn  # noqa: E402
from ldct_io.adversarial import AdversarialConfig, train_adversarial  # noqa: E402

#: Enough patches for the regimes to separate, few enough that the sweep is an afternoon and
#: not a week. The chosen weight is confirmed by the full run in liver_cnn.py.
SWEEP_CASES = 4
SWEEP_PATCHES = 400
SWEEP_EPOCHS = 12
DEFAULT_WEIGHTS = (0.0, 0.01, 0.05, 0.2, 1.0, 5.0)


def fine_detail(stack: np.ndarray) -> np.ndarray:
    """What a one-pixel Gaussian does not keep: the texture band."""
    from scipy.ndimage import gaussian_filter

    smoothed = np.stack([gaussian_filter(plane, 1.0, mode="nearest") for plane in stack])
    return stack - smoothed


def diagnostics(output: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Texture against the target's texture, and whether the output is still this image."""
    detail_out = fine_detail(output)
    detail_ref = fine_detail(target)
    sd_out = float(detail_out.std())
    sd_ref = float(detail_ref.std())
    flat_out = output.reshape(output.shape[0], -1)
    flat_ref = target.reshape(target.shape[0], -1)
    correlations = [
        float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]) for a, b in zip(flat_out, flat_ref)
    ]
    return {
        "texture_ratio": sd_out / sd_ref if sd_ref > 0 else float("nan"),
        "output_detail_sd": sd_out,
        "target_detail_sd": sd_ref,
        "registration": float(np.median(correlations)),
        "rmse": float(np.sqrt(np.mean((output - target) ** 2))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    # The first sweep showed why these have to be reachable. The pixelwise term is in units of
    # the input's own noise and sits near 3.5, while the least-squares adversarial term is
    # bounded near 0.25 when the critic is confused, so the objective is dominated by fidelity
    # at every weight that looked reasonable a priori. What matters is the ratio, and a critic
    # that learns fast enough to keep supplying gradient.
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=SWEEP_EPOCHS)
    parser.add_argument("--patches", type=int, default=SWEEP_PATCHES)
    parser.add_argument("--cases", type=int, default=SWEEP_CASES)
    args = parser.parse_args(argv)

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cases = TRAIN_CASES[: int(args.cases)]
    print(f"reading {len(cases)} cases for {args.patches} patches each", flush=True)
    x, y = training_pairs(cases, int(args.patches), seed=SEED)
    print(f"{x.shape[0]} patches of {x.shape[1]}x{x.shape[2]}", flush=True)

    # Held out from every run in the sweep, and the same for all of them.
    cut = int(0.85 * x.shape[0])
    x_eval, y_eval = x[cut:], y[cut:]

    preset = PRESETS["small"]
    rows: list[dict[str, Any]] = []
    for weight in args.weights:
        t0 = time.time()
        print(f"\nadv_weight = {weight}", flush=True)
        generator = build_cnn(preset["cnn"], seed=SEED)
        record = train_adversarial(
            generator,
            x,
            y,
            epochs=int(args.epochs),
            batch=preset["batch"],
            lr=preset["lr"],
            seed=SEED,
            device=device,
            config=AdversarialConfig(
                adv_weight=float(weight),
                mse_weight=float(args.mse_weight),
                **({} if args.critic_lr is None else {"discriminator_lr": float(args.critic_lr)}),
            ),
            log=lambda line: print(line, flush=True),
        )
        generator.eval()
        with torch.no_grad():
            produced = (
                generator(torch.from_numpy(x_eval).unsqueeze(1)).squeeze(1).numpy()
            )
        row = {
            "adv_weight": float(weight),
            "mse_weight": float(args.mse_weight),
            "critic_lr": record["objective"]["discriminator_lr"],
            "best_val_mse": record["best_val_mse"],
            "final_val_mse": record["final_val_mse"],
            "generator_sha256": record["generator_sha256"],
            "seconds": round(time.time() - t0, 1),
            **diagnostics(produced.astype(np.float64), y_eval.astype(np.float64)),
        }
        rows.append(row)
        print(
            f"  -> val MSE {row['final_val_mse']:.4f}  texture {row['texture_ratio']:.3f}  "
            f"registration {row['registration']:.3f}  ({row['seconds']:.0f}s)",
            flush=True,
        )

    # The unprocessed input, as the reference both failure modes are read against.
    baseline = diagnostics(x_eval.astype(np.float64), y_eval.astype(np.float64))
    out = OUTDIR / "adv_weight_sweep.json"
    out.write_text(
        json.dumps(
            {
                "what": "adversarial weight sweep: where the generator starts making texture",
                "cases": cases,
                "n_patches": int(x.shape[0]),
                "n_eval_patches": int(x_eval.shape[0]),
                "epochs": int(args.epochs),
                "mse_weight": float(args.mse_weight),
                "preset": "small",
                "seed": SEED,
                "device": device,
                "unprocessed": baseline,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nunprocessed: texture {baseline['texture_ratio']:.3f}, "
          f"registration {baseline['registration']:.3f}")
    print(f"wrote {out}")
    print(
        "\nRead it this way: the weight to use is the largest one whose registration is still "
        "high\nand whose texture ratio has come up towards one. A ratio near one with a fallen "
        "registration\nis a collapse, not a denoiser."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
