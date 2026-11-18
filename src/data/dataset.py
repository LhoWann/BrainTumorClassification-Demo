"""Membaca daftar gambar (tanpa duplikat), membuat Dataset/DataLoader PyTorch, dan membagi fold."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from src.config import Config

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_classes(folder: Path) -> list[str]:
    """Nama kelas = nama subfolder, diurutkan agar label selalu konsisten."""
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def list_images(
    folder: Path, classes: list[str], exclude: frozenset[str] = frozenset()
) -> tuple[list[Path], np.ndarray, set[str]]:
    """Kumpulkan path gambar dan label dari folder/<kelas>/<gambar>, tanpa gambar duplikat.

    Gambar dengan isi file identik (dicek lewat hash SHA-256) hanya diambil sekali. Tanpa langkah ini, gambar yang
    sama bisa masuk data train dan valid sekaligus, sehingga model terlihat bagus karena menghafal.

    Args:
        folder: Folder split, misalnya data/Training.
        classes: Daftar nama kelas; posisi di list menjadi label.
        exclude: Hash gambar yang tidak boleh diambil (misalnya gambar yang sudah ada di Testing).

    Returns:
        Daftar path gambar, array label, dan hash semua gambar yang diambil.
    """
    paths, labels, hashes = [], [], set()
    for label, name in enumerate(classes):
        for path in sorted((folder / name).iterdir()):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix.lower() not in IMG_EXTS or digest in hashes or digest in exclude:
                continue
            hashes.add(digest)
            paths.append(path)
            labels.append(label)
    return paths, np.array(labels), hashes


def load_dataset(data_dir: Path) -> tuple[list[str], tuple[list[Path], np.ndarray], tuple[list[Path], np.ndarray]]:
    """Baca kelas, daftar gambar Training dan Testing (duplikat dibuang, Testing tidak boleh bocor ke Training).

    Returns:
        Nama kelas, (path, label) Training, dan (path, label) Testing.
    """
    if not (data_dir / "Training").is_dir() or not (data_dir / "Testing").is_dir():
        raise FileNotFoundError(f"Ekstrak archive.zip ke {data_dir.resolve()} sehingga ada folder Training dan Testing")
    classes = find_classes(data_dir / "Training")
    test_paths, test_labels, test_hashes = list_images(data_dir / "Testing", classes)
    train_paths, train_labels, _ = list_images(data_dir / "Training", classes, exclude=frozenset(test_hashes))
    print(f"Kelas: {classes} | Training: {len(train_paths)} | Testing: {len(test_paths)} gambar unik")
    return classes, (train_paths, train_labels), (test_paths, test_labels)


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


def make_loader(dataset: Dataset, cfg: Config, train: bool = False) -> DataLoader:
    """DataLoader dari config. Data train diacak dan batch terakhir yang tidak penuh dibuang (aman untuk BatchNorm).

    Tanpa worker tambahan: gambar sudah ada di RAM, dan di Windows tiap worker memuat ulang torch (boros RAM).
    """
    return DataLoader(dataset, cfg.batch_size, shuffle=train, drop_last=train, pin_memory=cfg.device == "cuda")