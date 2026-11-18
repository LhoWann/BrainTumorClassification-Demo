"""Optimizer Muon untuk bobot hidden layer (termasuk filter konvolusi 4D).

Muon = momentum SGD yang arah update-nya "diortogonalkan" dengan iterasi Newton-Schulz, sehingga semua arah di
matriks bobot ikut belajar dengan kecepatan seimbang. torch.optim.Muon hanya menerima bobot 2D, jadi versi kecil ini
meratakan filter konvolusi [out, in, k, k] menjadi matriks [out, in*k*k] seperti implementasi asli Keller Jordan.
Muon hanya untuk hidden layer; bias, BatchNorm, dan layer output tetap memakai AdamW.
"""

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def newton_schulz(g: Tensor, steps: int = 5) -> Tensor:
    """Dekati matriks ortogonal terdekat dari g (semua singular value didorong mendekati 1)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7)
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """Muon: momentum Nesterov -> ortogonalisasi Newton-Schulz -> update bobot."""

    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0) -> None:
        """Simpan hyperparameter.

        Args:
            params: Bobot hidden layer berdimensi >= 2.
            lr: Learning rate (umumnya jauh lebih besar dari AdamW, mis. 0.02).
            momentum: Faktor momentum.
            weight_decay: Weight decay terpisah (decoupled) seperti AdamW.
        """
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p))
                buf.lerp_(p.grad, 1 - beta)
                matrix = p.grad.lerp(buf, beta).reshape(len(p), -1)
                update = newton_schulz(matrix) * max(1.0, matrix.size(0) / matrix.size(1)) ** 0.5
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-lr)