"""Semua pengaturan eksperimen dikumpulkan di satu tempat."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Hyperparameter dan lokasi folder.

    Attributes:
        models: Model yang dilatih dan dibandingkan (lihat src/models/).
        data_dir: Folder dataset yang berisi subfolder Training/ dan Testing/.
        output_dir: Folder untuk gambar hasil training/testing dan tabel metrik.
        img_size: Ukuran sisi gambar setelah resize (piksel).
        batch_size: Jumlah gambar per batch yang masuk GPU sekaligus.
        accum_steps: Gradient accumulation; batch efektif = batch_size x accum_steps.
        epochs: Jumlah epoch training per fold.
        lr: Learning rate optimizer AdamW.
        n_folds: Jumlah fold cross validation.
        device: "cuda" (GPU, mixed precision bfloat16) atau "cpu".
        pretrained: Pakai bobot ImageNet untuk EfficientNet-B0 (model CNN scratch selalu dari nol).
        seed: Seed agar hasil bisa diulang.
    """

    models: tuple[str, ...] = ("cnn_scratch", "cnn_modified", "efficientnet_b0")
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    img_size: int = 224
    batch_size: int = 16
    accum_steps: int = 2
    epochs: int = 10
    lr: float = 3e-4
    n_folds: int = 5
    device: str = "cuda"
    pretrained: bool = True
    seed: int = 42