"""Membaca daftar gambar, membuat Dataset PyTorch, dan membagi fold."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_classes(folder: Path) -> list[str]:
    """Nama kelas = nama subfolder, diurutkan agar label selalu konsisten."""
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def list_images(folder: Path, classes: list[str]) -> tuple[list[Path], np.ndarray]:
    """Kumpulkan path gambar dan label angkanya dari folder/<kelas>/<gambar>.

    Args:
        folder: Folder split, misalnya data/Training.
        classes: Daftar nama kelas; posisi di list menjadi label.

    Returns:
        Daftar path gambar dan array label.
    """
    paths, labels = [], []
    for label, name in enumerate(classes):
        for path in sorted((folder / name).iterdir()):
            if path.suffix.lower() in IMG_EXTS:
                paths.append(path)
                labels.append(label)
    return paths, np.array(labels)


def make_folds(labels: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bagi data Training menjadi fold train/valid dengan proporsi kelas yang sama (stratified)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


class ImageDataset(Dataset):
    """Dataset dari gambar yang sudah dimuat di RAM (lihat load_images), lalu diberi transform."""

    def __init__(self, images: list[Image.Image], labels: np.ndarray, transform: Callable) -> None:
        """Simpan daftar gambar.

        Args:
            images: Gambar hasil load_images.
            labels: Label angka setiap gambar.
            transform: Fungsi PIL Image -> Tensor (augmentasi/normalisasi).
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[idx]), int(self.labels[idx])