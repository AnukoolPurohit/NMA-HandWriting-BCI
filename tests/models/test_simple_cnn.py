import torch
import random
import pytest
from tests.macros import macro_image_sizes
from handwritingBCI.models.simple_cnn import Downsample, ConvBlock, SimpleCNN


class TestSimpleCNN:
    def test_output_size(self):
        in_channels = random.randint(8, 128)
        out_channels = random.randint(8, 128)
        height = random.randint(10, 200)
        width = random.randint(10, 200)
        batch_size = random.randint(10, 64)
        fc_dims = random.randint(100, 200)
        num_classes = random.randint(4, 100)
        input_shape = (in_channels, height, width)
        input_sample = torch.rand((batch_size, in_channels, height, width))
        conv_net = SimpleCNN(input_shape, out_channels,
                             fc_dims, num_classes)

        result = conv_net(input_sample)

        assert result.shape == (batch_size, num_classes)
        return


class TestConvBlock:
    def test_output_size(self):
        in_channels = random.randint(8, 128)
        out_channels = random.randint(8, 128)
        height = random.randint(10, 200)
        width = random.randint(10, 200)
        batch_size = random.randint(10, 64)

        input_sample = torch.rand((batch_size, in_channels, height, width))
        conv_block = ConvBlock(in_channels, out_channels)

        result = conv_block(input_sample)

        assert result.shape == (batch_size, out_channels, height, width)


class TestDownsample:
    @pytest.mark.parametrize("input_size,expected", macro_image_sizes)
    def test_output_shape(self, input_size, expected):
        sample_data = torch.rand((64, 1, input_size[0], input_size[1]))
        downsample = Downsample(1)
        result = downsample(sample_data)
        assert result.shape == (64, 1, expected[0], expected[1])
        return



