"""Utility class for loss function"""
from torchvision import models
from torch import nn, tensor
from torchvision.transforms import Normalize


class TruncatedVGG(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution
    (after activation) before the i-th maxpooling layer within the VGG19 network', as defined in the paper.

    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss. (perceptual loss)
    """

    def __init__(self, i: int = 5, j: int = 4):
        """

        Parameters
        ----------
        i: int, default 5
            index of the maxpooling layer
        j: int, default 4
            index of the convolution layer

        Raises
        ------
        ValueError
            If i or j indexes are not possible choices on the VGG network.

        """
        super().__init__()

        # Load the pre-trained VGG19 available in torchvision.
        vgg19 = models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        if maxpool_counter != i - 1 or conv_counter != j:
            raise ValueError(f"i = {i} or j = {j} are not valid choices for the VGG19!")

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

        # VGG model parameters should not be updated.
        for param in self.truncated_vgg19.parameters():
            param.requires_grad = False

        # ImageNET normalization for VGG input.
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input: tensor):
        """
        Forward  propagation.

        Parameters
        ----------
        input: tensor
            High-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor).

        Returns
        -------
        tensor
            The specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h).

        """
        input = self.normalize(input)
        return self.truncated_vgg19(input)
