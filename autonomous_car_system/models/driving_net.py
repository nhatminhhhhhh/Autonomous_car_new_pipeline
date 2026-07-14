import torch
import torch.nn as nn

class DrivingNet(nn.Module):
    def __init__(self, input_height=66, input_width=200, in_channels=3, output_dim=1):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),

            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_height, input_width)
            dummy_out = self.conv_layers(dummy)
            flattened_size = dummy_out.view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(50, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, output_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
