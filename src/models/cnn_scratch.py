"""Model 1: CNN sederhana yang dilatih dari nol."""

from torch import Tensor, nn

from src.models.blocks import conv_block


class SimpleCNN(nn.Module):
    """3 blok konvolusi (16-32-64 channel) lalu Flatten + satu layer Linear."""

    def __init__(self, num_classes: int, img_size: int) -> None:
        """Susun layer.

        Args:
            num_classes: Jumlah kelas output.
            img_size: Ukuran sisi gambar input, dipakai untuk menghitung ukuran Flatten.
        """
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 16), conv_block(16, 32), conv_block(32, 64))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * (img_size // 8) ** 2, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))