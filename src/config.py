"""Semua pengaturan eksperimen dikumpulkan di satu tempat."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Hyperparameter dan lokasi folder.

    Nilai lr, muon_lr, weight_decay, label_smoothing, dan dropout di bawah adalah default. Jika file
    outputs/tuning/best_params.json ada (hasil `python -m src.tuning`), main.py memakai nilai dari file itu per model.

    Attributes:
        models: Model yang dilatih dan dibandingkan (lihat src/models/).
        data_dir: Folder dataset yang berisi subfolder Training/ dan Testing/.
        output_dir: Folder untuk gambar hasil training/testing dan tabel metrik.
        img_size: Ukuran sisi gambar setelah resize (piksel).
        batch_size: Jumlah gambar per batch yang masuk GPU sekaligus.
        accum_steps: Gradient accumulation; batch efektif = batch_size x accum_steps.
        epochs: Batas maksimum epoch per fold; biasanya berhenti lebih awal karena early stopping.
        patience: Early stopping: berhenti jika val loss tidak membaik selama sekian epoch.
        lr: Learning rate awal AdamW (dibagi 2 otomatis jika val loss stagnan 2 epoch).
        muon_models: Model yang filter konvolusi hidden layer-nya dilatih dengan Muon (sisanya AdamW).
        muon_lr: Learning rate awal Muon (umumnya jauh lebih besar dari AdamW).
        weight_decay: Regularisasi L2 AdamW; menahan bobot agar tidak terlalu besar (mengurangi menghafal).
        label_smoothing: Target tidak 100% yakin (mis. 0.1), model tidak dipaksa terlalu percaya diri.
        dropout: Peluang neuron dimatikan saat training (cnn_modified dan efficientnet_b0).
        n_folds: Jumlah fold cross validation.
        pretrained: Pakai bobot ImageNet untuk EfficientNet-B0 (model CNN scratch selalu dari nol).
        device: "cuda" (GPU) atau "cpu".
        seed: Seed agar hasil bisa diulang.
        tune_trials: Jumlah trial Optuna per model.
        tune_epochs: Batas epoch per trial Optuna (lebih pendek dari epochs agar tuning cepat).
    """

    models: tuple[str, ...] = ("cnn_scratch", "cnn_modified", "efficientnet_b0")
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    img_size: int = 224
    batch_size: int = 16
    accum_steps: int = 2
    epochs: int = 40
    patience: int = 6
    lr: float = 3e-4
    muon_models: tuple[str, ...] = ("cnn_modified",)
    muon_lr: float = 0.02
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    dropout: float = 0.3
    n_folds: int = 5
    pretrained: bool = True
    device: str = "cuda"
    seed: int = 42
    tune_trials: int = 20
    tune_epochs: int = 12