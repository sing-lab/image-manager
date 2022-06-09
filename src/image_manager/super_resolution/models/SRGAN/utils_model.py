"""Utility classes for SRGAN model."""
from torch import nn,  tensor
from typing import Optional


class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, batch normalization and activation layers.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, batch_norm: bool = False,
                 activation: Optional[str] = None):
        """

        Parameters
        ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        kernel_size: int
            The kernel size.
        stride: int, default 1
            The stride.
        batch_norm: bool, default False
            Whether to include a batch normalization layer.
        activation: Optional[str], default None
            The type of activation to use, should be in 'LeakyReLu', 'Prelu', 'Tanh'.

        Raises
        ------
        ValueError
            If the activation function is not in 'tanh', 'prelu'.

        """
        super().__init__()

        # A convolutional layer
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)

        # A container that will hold the layers in this convolutional block
        layers = [conv_layer]

        # An optional batch normalization layer
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An optional activation layer, if wanted
        if activation is not None:
            if activation.lower() == "prelu":
                layers.append(nn.PReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "leakyrelu":
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Activation should be either 'leakyrelu', 'prelu', or 'tanh' but is '{activation}'.")

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input: tensor):
        """
        Forward  propagation.

        Parameters
        ----------
        input: tensor
            input images, a tensor of size (N, in_channels, w, h)

        Returns
        -------
        tensor
            Output images, a tensor of size (N, out_channels, w, h)

        """
        return self.conv_block(input)  # (N, out_channels, w, h)


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, kernel_size: int = 3, n_channels: int = 64):
        """

        Parameters
        ----------
        kernel_size: int, default 3
            The kernel size.
        n_channels: int, default 64
            The number of input and output channels, identical because the input must be added to the output with
            skip-connections.
        """
        super().__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input: tensor):
        """
        Forward propagation.

        Parameters
        ----------
        input: tensor
            Input images, a tensor of size (N, n_channels, w, h).

        Returns
        -------
        tensor
            Output images, a tensor of size (N, n_channels, w, h).
        """
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)

        # Skip-connection
        return output + input


class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size: int = 3, n_channels: int = 64, scaling_factor: int = 2):
        """

        Parameters
        ----------
        kernel_size: int, default 3
            The kernel size.
        n_channels: int, default 64
            The number of input and output channels.
        scaling_factor: int, default 2
           The factor to scale input images by (along both dimensions).
        """
        super().__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle
        # and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling
        # factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.activation = nn.PReLU()

    def forward(self, input: tensor):
        """
        Forward propagation.

        Parameters
        ----------
        input: tensor
            Input images, a tensor of size (N, n_channels, w, h).

        Returns
        -------
        tensor
            Scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        return self.activation(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
