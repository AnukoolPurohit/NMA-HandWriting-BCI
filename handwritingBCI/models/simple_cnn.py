import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged),
    followed by a ReLU .
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        r1 = self.relu(c1)

        c2 = self.conv2(r1)
        c2 = self.bn2(c2)
        r2 = self.relu(c2)
        return r2


class Downsample(nn.Module):
    """
        Simple 2x2 conv with stride 2 and 0 padding to downsample instead of a maxpool
    """

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, input_channels,
                              kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        output = self.relu(x)
        return output


class SimpleCNN(nn.Module):
    def __init__(self, input_shape: tuple, output_channels: int,
                 fc_dims: int, num_classes: int) -> None:
        super().__init__()
        assert len(input_shape) == 3
        input_channels = input_shape[0]

        self.conv_block1 = ConvBlock(input_channels, output_channels)
        self.conv_block2 = ConvBlock(output_channels, output_channels)
        self.downsample = Downsample(output_channels)

        input_fc_dims = output_channels * (input_shape[1] // 2) * (input_shape[2] // 2)

        self.fc1 = nn.Linear(input_fc_dims, fc_dims)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dims, num_classes)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.downsample(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
