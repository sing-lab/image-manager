"""Discriminator model."""
from torch import nn, tensor
from utils_model import ConvolutionalBlock


class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    CNN which inputs the image, and aims at classifying whether it is a real image or a false image.
    """

    def __init__(self, kernel_size: int = 3, n_channels: int = 64, n_conv_blocks: int = 8, fc_size: int = 1024):
        """

        Parameters
        ----------
        kernel_size: int, default 3
            The kernel size in all convolutional blocks.
        n_channels: int, default 64
            The number of output channels in the first convolutional block, after which it is doubled in every 2nd block
            thereafter.
        n_conv_blocks: int, default 8
            The number of convolutional blocks.
        fc_size: int, default 1024
            The size of the first fully connected layer.
        """
        super().__init__()

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization

        conv_blocks = []
        in_channels = 3  # Number of input channels for the first convolutional block.

        for i in range(n_conv_blocks):

            # out_channels and stride
            if not i % 2:
                stride = 1
                if not i:
                    out_channels = n_channels
                else:
                    out_channels = in_channels * 2
            else:
                stride = 2
                out_channels = in_channels

            # batch_norm
            if not i:
                batch_norm = False
            else:
                batch_norm = True

            conv_block = ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, batch_norm=batch_norm, activation='LeakyReLu')
            conv_blocks.append(conv_block)
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.activation = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

    def forward(self, input: tensor):
        """
        Forward propagation.

        Parameters
        ----------
        input: tensor
            High-resolution or super-resolution images which must be classified as such, a tensor of size
            (N, 3, w * scaling factor, h * scaling factor)

        Returns
        -------
        tensor
             A score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = input.size(0)
        output = self.conv_blocks(input)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.activation(output)
        logit = self.fc2(output)

        return logit
