"""Preprocessing gambar: crop tepi hitam + resize (sekali di awal), lalu augmentasi (khusus train) dan normalisasi."""

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropBlackBorder:
    """Buang tepi hitam di sekeliling citra MRI agar model fokus ke area otak."""

    def __init__(self, threshold: int = 10) -> None:
        """Atur ambang piksel.

        Args:
            threshold: Piksel dengan intensitas <= nilai ini dianggap latar hitam.
        """
        self.threshold = threshold

    def __call__(self, image: Image.Image) -> Image.Image:
        mask = np.asarray(image.convert("L")) > self.threshold
        if not mask.any():
            return image
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        return image.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def load_images(paths: list[Path], img_size: int) -> list[Image.Image]:
    """Buka semua gambar sekali saja: RGB -> crop tepi hitam -> resize. Hasilnya disimpan di RAM agar training cepat.

    Args:
        paths: Path setiap gambar.
        img_size: Ukuran sisi gambar setelah resize.

    Returns:
        Gambar kecil yang siap diaugmentasi.
    """
    crop, images = CropBlackBorder(), []
    for path in tqdm(paths, desc="baca gambar", leave=False):
        with Image.open(path) as image:
            images.append(crop(image.convert("RGB")).resize((img_size, img_size)))
    return images


def train_transform() -> transforms.Compose:
    """Augmentasi acak untuk data train (memperbanyak variasi gambar) + normalisasi."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> transforms.Compose:
    """Tanpa augmentasi untuk data valid dan test, hanya normalisasi."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])