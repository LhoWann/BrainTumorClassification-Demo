"""Model 3: EfficientNet-B0 pretrained ImageNet (transfer learning)."""

from torch import nn
from torchvision import models


def efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 dengan layer terakhir diganti sesuai jumlah kelas.

    Args:
        num_classes: Jumlah kelas output.
        pretrained: Muat bobot ImageNet (perlu internet saat pertama kali).
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model