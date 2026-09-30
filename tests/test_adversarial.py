"""The adversarial arm has to be adversarial, and has to stay inside the bound's premise.

Two classes of test. The first is ordinary: shapes, seeds, validation. The second is the one
that matters for the paper — a generator whose adversarial weight is doing nothing is an MSE
network with extra steps, and would quietly turn the experiment into a null. So the tests
compare the two objectives rather than checking that each runs.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ldct_io.adversarial import (  # noqa: E402
    AdversarialConfig,
    DiscriminatorConfig,
    build_discriminator,
    train_adversarial,
    weights_sha256,
)


def _tiny_generator(seed: int = 0):
    """A residual denoiser small enough to train in a test, built the same way twice."""
    torch.manual_seed(seed)
    body = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(8, 8, 3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(8, 1, 3, padding=1),
    )

    class Residual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = body

        def forward(self, x):
            return x - self.body(x)

    return Residual()


def _pairs(n: int = 64, size: int = 32, seed: int = 1):
    """Noisy/clean pairs with texture in the target, so appearance and fidelity can differ."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, size=(n, size, size))
    for k in range(n):  # a few blobs, so "looks like the target" is not "is flat"
        for _ in range(3):
            cy, cx = rng.integers(6, size - 6, size=2)
            yy, xx = np.ogrid[:size, :size]
            clean[k] += 2.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
    noisy = clean + rng.normal(0.0, 1.5, size=clean.shape)
    return noisy.astype(np.float32), clean.astype(np.float32)


def test_the_critic_maps_a_batch_to_one_score_each():
    critic = build_discriminator(DiscriminatorConfig(depth=2, width=8))
    scores = critic(torch.zeros(5, 1, 32, 32))
    assert scores.shape == (5, 1)


def test_the_critic_is_seeded():
    a = build_discriminator(DiscriminatorConfig(depth=2, width=8), seed=7)
    b = build_discriminator(DiscriminatorConfig(depth=2, width=8), seed=7)
    c = build_discriminator(DiscriminatorConfig(depth=2, width=8), seed=8)
    assert weights_sha256(a) == weights_sha256(b)
    assert weights_sha256(a) != weights_sha256(c)


@pytest.mark.parametrize("depth,kernel", [(0, 3), (2, 4)])
def test_the_critic_refuses_a_shape_it_cannot_build(depth, kernel):
    with pytest.raises(ValueError):
        build_discriminator(DiscriminatorConfig(depth=depth, kernel_size=kernel))


def test_training_reports_what_it_did():
    noisy, clean = _pairs()
    record = train_adversarial(
        _tiny_generator(), noisy, clean, epochs=2, batch=16, lr=1e-3, seed=0
    )
    assert len(record["history"]["val_mse"]) == 2
    assert record["n_train_patches"] + record["n_val_patches"] == noisy.shape[0]
    assert record["generator_parameters"] > 0 and record["critic_parameters"] > 0
    assert len(record["generator_sha256"]) == 64
    assert record["objective"]["adv_weight"] == AdversarialConfig().adv_weight


def test_it_refuses_mismatched_or_wrongly_shaped_data():
    noisy, clean = _pairs(n=16, size=16)
    with pytest.raises(ValueError):
        train_adversarial(_tiny_generator(), noisy, clean[:8], epochs=1, batch=4, lr=1e-3)
    with pytest.raises(ValueError):
        train_adversarial(_tiny_generator(), noisy[0], clean[0], epochs=1, batch=4, lr=1e-3)


def test_the_adversarial_term_changes_the_network():
    """Zero adversarial weight and the default must not produce the same weights.

    If they did, the arm would be an MSE network under another name and every downstream
    comparison in the paper would be between a denoiser and itself.
    """
    noisy, clean = _pairs()
    off = train_adversarial(
        _tiny_generator(), noisy, clean, epochs=3, batch=16, lr=1e-3, seed=0,
        config=AdversarialConfig(adv_weight=0.0),
    )
    on = train_adversarial(
        _tiny_generator(), noisy, clean, epochs=3, batch=16, lr=1e-3, seed=0,
        config=AdversarialConfig(adv_weight=0.5),
    )
    assert off["generator_sha256"] != on["generator_sha256"]


def test_the_adversarial_arm_pays_for_it_in_mse():
    """Training for appearance costs fidelity, which is the premise of the whole comparison.

    The adversarial generator is not trying to minimise MSE alone, so its validation MSE must
    not be better than the one trained on MSE alone. If it were, the adversarial term would be
    acting as a regulariser rather than as a pull towards plausible texture, and "fidelity rose
    while the task fell" would have no mechanism behind it.
    """
    noisy, clean = _pairs(n=128)
    fidelity = train_adversarial(
        _tiny_generator(), noisy, clean, epochs=6, batch=16, lr=1e-3, seed=0,
        config=AdversarialConfig(adv_weight=0.0),
    )
    appearance = train_adversarial(
        _tiny_generator(), noisy, clean, epochs=6, batch=16, lr=1e-3, seed=0,
        config=AdversarialConfig(adv_weight=1.0),
    )
    assert appearance["best_val_mse"] >= fidelity["best_val_mse"]


def test_the_generator_sees_the_image_and_nothing_else():
    """The premise of the data-processing inequality, asserted rather than assumed.

    Two inputs that differ only in their pixels must give outputs that differ only through
    those pixels: the same image scored twice gives the same output, and no hidden state
    carries between calls. A generator that sampled a latent would fail this, and would not be
    covered by the bound the paper compares it against.
    """
    noisy, clean = _pairs(n=32)
    generator = _tiny_generator()
    train_adversarial(generator, noisy, clean, epochs=1, batch=8, lr=1e-3, seed=0)
    generator.eval()
    image = torch.from_numpy(noisy[:4]).unsqueeze(1)
    with torch.no_grad():
        first = generator(image)
        second = generator(image)
        again = generator(image.clone())
    assert torch.equal(first, second)
    assert torch.equal(first, again)
