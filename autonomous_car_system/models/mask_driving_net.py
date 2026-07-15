"""
MaskDrivingNet — Tiny CNN nhận đầu vào là road segmentation mask (1-channel)
thay vì ảnh gốc RGB. Nhẹ hơn DrivingNet, ít noise hơn vì mask đã trừu tượng hóa.

Input:  [B, 1, 128, 128]  — mask normalized (class index / num_classes)
Output: [B, 1]            — steering angle trong [-1.0, 1.0]
"""

import torch
import torch.nn as nn


class MaskDrivingNet(nn.Module):
    """
    Tiny CNN học cách lái xe từ mask phân đoạn đường.

    Kiến trúc:
        4 Conv block (stride=2) để downsample + trích đặc trưng
        → Global Average Pooling
        → 2 FC layers + Dropout
        → Tanh output ∈ [-1.0, +1.0]

    Tham số: ~80K (nhẹ hơn DrivingNet nhiều)
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_height: int = 128,
        input_width: int = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width

        # Encoder: 4 Conv block với stride=2 → downsample 128→64→32→16→8
        self.encoder = nn.Sequential(
            # Block 1: 128×128 → 64×64
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: 64×64 → 32×32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling: [B, 128, 8, 8] → [B, 128]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully-connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),                      # [B, 128]
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
            nn.Tanh(),                         # output ∈ [-1.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]  — mask đã normalize
        Returns:
            [B, 1]  — steering angle ∈ [-1.0, 1.0]
        """
        x = self.encoder(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    import torch

    model = MaskDrivingNet(in_channels=1, input_height=128, input_width=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MaskDrivingNet params: {total_params:,}")

    # Test forward pass
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
