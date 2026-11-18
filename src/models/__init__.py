"""Pilih model berdasarkan nama."""

from torch import nn

from src.models.cnn_modified import ModifiedCNN
from src.models.cnn_scratch import SimpleCNN
from src.models.efficientnet import efficientnet_b0

MODEL_NAMES = ("cnn_scratch", "cnn_modified", "efficientnet_b0")


def build_model(name: str, num_classes: int, img_size: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Buat model berdasarkan nama.

    Args:
        name: Salah satu dari MODEL_NAMES.
        num_classes: Jumlah kelas output.
        img_size: Ukuran sisi gambar input.
        pretrained: Muat bobot ImageNet untuk EfficientNet-B0.
        dropout: Dropout sebelum layer terakhir (tidak dipakai cnn_scratch, sengaja sebagai pembanding).

    Returns:
        Model siap dilatih.
    """
    if name == "cnn_scratch":
        return SimpleCNN(num_classes, img_size)
    if name == "cnn_modified":
        return ModifiedCNN(num_classes, dropout)
    if name == "efficientnet_b0":
        return efficientnet_b0(num_classes, pretrained, dropout)
    raise ValueError(f"Model tidak dikenal: {name}. Pilih dari {MODEL_NAMES}")