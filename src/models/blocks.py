"""Blok bangunan yang dipakai bersama oleh CNN scratch dan CNN modifikasi."""

from torch import nn


def conv_block(in_ch: int, out_ch: int, batch_norm: bool = False) -> nn.Sequential:
    """Conv 3x3 -> (BatchNorm) -> ReLU -> MaxPool 2x2 (ukuran gambar jadi setengah)."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)