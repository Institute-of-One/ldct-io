"""Train the same denoiser against appearance instead of against fidelity.

Why this exists
---------------
The ceiling argument says that no denoiser can raise the detectability of its own input,
because processing cannot add information about the hypothesis. A denoiser that appears to beat
that ceiling is therefore not recovering signal; it is supplying structure of its own. The
measurement that would catch it is already written — ``denoiq_core.redlamp.false_structure_rate``
counts lesion-shaped responses in images where the truth is a flat background, and
``contrast_recovery`` says how much of a real lesion survives — but it has only ever been aimed
at denoisers that cannot fabricate. A Gaussian filter, total variation, non-local means and an
MSE-trained residual CNN all regress towards the mean: they remove contrast, they do not invent
it, and the detector never fires.

This module supplies the arm that can fabricate, and changes **one thing** to get it. The
generator is the same architecture, trained on the same patches, for the same number of epochs,
from the same seed, with the same optimiser and learning rate as the MSE arm. Only the objective
differs: a least-squares adversarial term is added, so the network is rewarded for producing
images a discriminator cannot tell from full-dose ones rather than for being close to them
pixel by pixel. If fabrication follows, it follows from optimising appearance, and not from
capacity, data or training length, all of which are held fixed.

What is preserved
-----------------
The bound still applies. At inference the generator is a deterministic function of its input
image and of nothing else: it takes no noise vector, no class label, no lesion location and no
other trial's pixels, so ``H -> X -> Y`` holds and the data-processing inequality bounds it
exactly as it bounds a Gaussian filter. A generator that sampled from a latent would not be
covered by that argument, which is why this one does not.

Why least squares rather than the log loss
------------------------------------------
The saturating and non-saturating forms of the log loss both need care near the ends of the
discriminator's range, and their instabilities are a property of the loss rather than of the
question being asked here. The least-squares form (Mao et al., ICCV 2017) gives the same
adversarial pressure with bounded gradients, which keeps a run reproducible from a seed.

Determinism
-----------
Every seeded object is seeded from one integer, and the weights' SHA-256 is recorded so that a
rerun can be compared rather than assumed. Exact bitwise reproduction additionally requires
deterministic kernels; on CUDA that is not free, so what is claimed here is a seeded run and a
recorded hash, not bitwise equality across machines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the adversarial arm needs torch; install the project's optional 'cnn' extra"
        ) from exc
    return torch


@dataclass(frozen=True)
class DiscriminatorConfig:
    """The critic. Deliberately small.

    It is a patch discriminator: a stack of strided convolutions ending in a one-channel map,
    averaged to a scalar. It is small because its job is to notice that quarter-dose texture is
    not full-dose texture, not to be a classifier of record, and because a critic much stronger
    than the generator produces a generator that stops moving.

    Attributes
    ----------
    depth:
        Number of strided convolution blocks.
    width:
        Channels in the first block; each block doubles it, capped at ``max_width``.
    kernel_size:
        Convolution kernel, odd.
    negative_slope:
        LeakyReLU slope.
    max_width:
        Ceiling on the channel count, so ``depth`` does not blow the parameter count up.

    """

    depth: int = 3
    width: int = 32
    kernel_size: int = 3
    negative_slope: float = 0.2
    max_width: int = 128

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record of the critic's shape."""
        return {
            "depth": int(self.depth),
            "width": int(self.width),
            "kernel_size": int(self.kernel_size),
            "negative_slope": float(self.negative_slope),
            "max_width": int(self.max_width),
        }


@dataclass(frozen=True)
class AdversarialConfig:
    """The objective, and nothing about the generator.

    Attributes
    ----------
    mse_weight, adv_weight:
        The generator minimises ``mse_weight * MSE + adv_weight * adversarial``. The default
        keeps the MSE term present so the output stays registered to its input — a pure
        adversarial denoiser is free to return any plausible liver, which is a different
        experiment — while making the adversarial term large enough to change what is produced.
    discriminator_lr:
        The critic's learning rate. Separate from the generator's because the two are not
        solving the same problem.
    discriminator:
        Shape of the critic.
    label_real, label_fake:
        Least-squares targets.

    """

    mse_weight: float = 1.0
    adv_weight: float = 0.05
    discriminator_lr: float = 1e-4
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    label_real: float = 1.0
    label_fake: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record of the objective."""
        return {
            "loss": "least-squares adversarial (Mao et al. 2017) plus pixelwise MSE",
            "mse_weight": float(self.mse_weight),
            "adv_weight": float(self.adv_weight),
            "discriminator_lr": float(self.discriminator_lr),
            "discriminator": self.discriminator.to_dict(),
            "label_real": float(self.label_real),
            "label_fake": float(self.label_fake),
        }


def build_discriminator(config: DiscriminatorConfig, *, seed: int = 0) -> Any:
    """A patch critic, seeded.

    Parameters
    ----------
    config:
        Shape of the critic.
    seed:
        Seeds the weight initialisation, so two runs of the same experiment start from the
        same critic as well as the same generator.

    Returns
    -------
    torch.nn.Module
        Maps ``(n, 1, h, w)`` to ``(n, 1)``: the mean of a one-channel patch map.

    """
    torch = _require_torch()
    nn = torch.nn

    if config.depth < 1:
        raise ValueError(f"depth must be at least 1, got {config.depth}")
    if config.kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {config.kernel_size}")

    torch.manual_seed(int(seed))
    padding = config.kernel_size // 2
    layers: list[Any] = []
    in_channels = 1
    channels = int(config.width)
    for block in range(int(config.depth)):
        layers.append(
            nn.Conv2d(in_channels, channels, config.kernel_size, stride=2, padding=padding)
        )
        # No normalisation in the first block: it would remove the very offset that
        # distinguishes a quarter-dose patch from a full-dose one.
        if block > 0:
            layers.append(nn.GroupNorm(1, channels))
        layers.append(nn.LeakyReLU(config.negative_slope, inplace=True))
        in_channels = channels
        channels = min(channels * 2, int(config.max_width))
    layers.append(nn.Conv2d(in_channels, 1, config.kernel_size, stride=1, padding=padding))

    # torch is an optional extra, imported at call time by _require_torch, so `nn` is a local
    # name and the type checker -- which runs without torch installed -- cannot resolve the base
    # class. What it is reporting is the absence of an optional dependency, not a mistake.
    class PatchCritic(nn.Module):  # type: ignore[name-defined, misc]
        """Strided convolutions to a one-channel map, averaged."""

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            """Score a batch of images, one scalar each."""
            return self.body(x).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

    return PatchCritic()


def weights_sha256(model: Any) -> str:
    """SHA-256 over the model's parameters, in name order.

    A run is reported with this rather than with a claim of determinism, so that a rerun can be
    compared against it.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_adversarial(
    generator: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    config: AdversarialConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    val_fraction: float = 0.15,
    log: Any = None,
) -> dict[str, Any]:
    """Train ``generator`` against appearance, in place, and report what happened.

    The generator is trained, not built, here: it arrives already constructed by whatever
    builds the MSE arm's network, so the two arms cannot differ in architecture by accident.

    Parameters
    ----------
    generator:
        The network to train, modified in place and left on the CPU at the end.
    x, y:
        Noisy inputs and their full-dose targets, shape ``(n, h, w)``, already normalised the
        way inference will normalise them.
    epochs, batch, lr:
        The generator's schedule. Pass the MSE arm's values to keep the comparison controlled.
    config:
        The objective. ``None`` uses the defaults.
    seed:
        Seeds the critic, the shuffling and torch's global generator.
    device:
        ``"cpu"`` or ``"cuda"``.
    val_fraction:
        Held-out fraction of the patches, used only to report a validation MSE that is
        comparable with the MSE arm's.
    log:
        Optional callable taking a string, for progress.

    Returns
    -------
    dict
        ``val_mse`` per epoch and its best, the generator and critic losses per epoch, the
        parameter counts, the weights' SHA-256 and the configuration, all machine-readable.

    """
    torch = _require_torch()
    if x.shape != y.shape:
        raise ValueError(f"inputs and targets must match, got {x.shape} and {y.shape}")
    if x.ndim != 3:
        raise ValueError(f"expected a stack (n, h, w), got shape {x.shape}")
    settings = config or AdversarialConfig()

    torch.manual_seed(int(seed))
    critic = build_discriminator(settings.discriminator, seed=int(seed)).to(device)
    generator = generator.to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=float(lr))
    opt_d = torch.optim.Adam(critic.parameters(), lr=float(settings.discriminator_lr))
    mse = torch.nn.MSELoss()

    n = int(x.shape[0])
    cut = int((1.0 - float(val_fraction)) * n)
    if cut < 1 or cut >= n:
        raise ValueError(f"val_fraction leaves no split for n={n}")
    xt = torch.from_numpy(np.ascontiguousarray(x[:cut])).unsqueeze(1).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(y[:cut])).unsqueeze(1).to(device)
    xv = torch.from_numpy(np.ascontiguousarray(x[cut:])).unsqueeze(1).to(device)
    yv = torch.from_numpy(np.ascontiguousarray(y[cut:])).unsqueeze(1).to(device)

    shuffler = torch.Generator().manual_seed(int(seed))
    real = float(settings.label_real)
    fake = float(settings.label_fake)
    history: dict[str, list[float]] = {"generator": [], "critic": [], "val_mse": []}

    for epoch in range(int(epochs)):
        generator.train()
        critic.train()
        order = torch.randperm(cut, generator=shuffler).to(device)
        running_g = running_d = 0.0
        for start in range(0, cut, int(batch)):
            index = order[start : start + int(batch)]
            if index.numel() == 0:  # pragma: no cover - defensive
                continue
            noisy, clean = xt[index], yt[index]

            # The critic first, on the generator's current output detached from its graph.
            produced = generator(noisy)
            opt_d.zero_grad(set_to_none=True)
            loss_d = 0.5 * (
                ((critic(clean) - real) ** 2).mean()
                + ((critic(produced.detach()) - fake) ** 2).mean()
            )
            loss_d.backward()
            opt_d.step()

            # Then the generator, rewarded for a critic score near `real`.
            opt_g.zero_grad(set_to_none=True)
            loss_g = (
                float(settings.mse_weight) * mse(produced, clean)
                + float(settings.adv_weight) * 0.5 * ((critic(produced) - real) ** 2).mean()
            )
            loss_g.backward()
            opt_g.step()

            running_g += float(loss_g.detach()) * index.numel()
            running_d += float(loss_d.detach()) * index.numel()

        generator.eval()
        with torch.no_grad():
            val = float(
                sum(
                    float(mse(generator(xv[i : i + int(batch)]), yv[i : i + int(batch)]))
                    * xv[i : i + int(batch)].shape[0]
                    for i in range(0, xv.shape[0], int(batch))
                )
                / xv.shape[0]
            )
        history["generator"].append(running_g / cut)
        history["critic"].append(running_d / cut)
        history["val_mse"].append(val)
        if log is not None:
            log(
                f"  epoch {epoch + 1:3d}/{epochs}  G {running_g / cut:.5f}  "
                f"D {running_d / cut:.5f}  val MSE {val:.5f}"
            )

    generator = generator.cpu()
    return {
        "objective": settings.to_dict(),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch": int(batch),
        "generator_lr": float(lr),
        "val_fraction": float(val_fraction),
        "n_train_patches": cut,
        "n_val_patches": n - cut,
        "generator_parameters": int(sum(p.numel() for p in generator.parameters())),
        "critic_parameters": int(sum(p.numel() for p in critic.parameters())),
        "history": history,
        "best_val_mse": float(min(history["val_mse"])) if history["val_mse"] else float("nan"),
        "final_val_mse": float(history["val_mse"][-1]) if history["val_mse"] else float("nan"),
        "generator_sha256": weights_sha256(generator),
        "critic_sha256": weights_sha256(critic.cpu()),
    }
