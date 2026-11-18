"""Model 3: EfficientNet-B0 pretrained ImageNet (transfer learning, semua layer ikut dilatih)."""

from torch import nn
from torchvision import models


def efficientnet_b0(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    """EfficientNet-B0 dengan kepala klasifikasi baru: Dropout -> Linear(1280 -> jumlah kelas).

    Args:
        num_classes: Jumlah kelas output.
        pretrained: Muat bobot ImageNet (perlu internet saat pertama kali).
        dropout: Peluang neuron dimatikan sebelum layer Linear.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, num_classes))
    return model