"""Model 2: modifikasi SimpleCNN agar lebih stabil dan tidak mudah overfitting."""

from torch import Tensor, nn

from src.models.blocks import conv_block


class ModifiedCNN(nn.Module):
    """Perubahan dari SimpleCNN: +BatchNorm, +1 blok (32-64-128-256), Global Average Pooling, dan Dropout."""

    def __init__(self, num_classes: int) -> None:
        """Susun layer.

        Args:
            num_classes: Jumlah kelas output.
        """
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, batch_norm=True),
            conv_block(32, 64, batch_norm=True),
            conv_block(64, 128, batch_norm=True),
            conv_block(128, 256, batch_norm=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))