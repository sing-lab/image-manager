"""Generator model."""

import math

from torch import nn, tensor

from models.SRGAN.utils_model import ConvolutionalBlock, ResidualBlock, SubPixelConvolutionalBlock


class Generator(nn.Module):
    """ The generator model."""

    def __init__(self, large_kernel_size: int = 9, small_kernel_size: int = 3, n_channels: int = 64, n_blocks: int = 16,
                 scaling_factor: int = 4):
        """

        Parameters
        ----------
        large_kernel_size: int, default 9
            The kernel size of the first and last convolutions which transform the inputs and outputs.
        small_kernel_size: int, default 3
            The kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional
            blocks.
        n_channels: int, default 64
            The number of channels in-between, i.e. the input and output channels for the residual and subpixel
            convolutional blocks.
        n_blocks: int, default 16
            The number of residual blocks.
        scaling_factor: int, default 4
            The factor to scale input images by (along both dimensions) in the subpixel convolutional block.
        """
        super().__init__()

        # First convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # Sequence of n_blocks residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels)
                                               for i in range(n_blocks)])

        # Second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size, batch_norm=True, activation=None)

        # Upscaling using sub-pixel convolutions
        n_subpixel_conv_blocks = int(math.log2(scaling_factor))
        self.upscaling_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size,
                                         n_channels=n_channels,
                                         scaling_factor=2) for i in range(n_subpixel_conv_blocks)])

        # Last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, input: tensor):
        """
        Forward propagation.

        Parameters
        ----------
        input: tensor
            Low-resolution input images, a tensor of size (N, 3, w, h).

        Returns
        -------
        tensor
             Super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor).
        """
        output = self.conv_block1(input)
        residual = output.clone()
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = self.upscaling_blocks(residual + output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)

        return (output + 1.) / 2.  # Scale output from [-1, 1] to [0, 1], as we use 'Prelu' on previous layers.
