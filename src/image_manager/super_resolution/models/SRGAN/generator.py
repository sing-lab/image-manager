"""Generator model."""
import math
import os
from datetime import datetime
from time import time
from typing import Tuple, List, Dict, Union
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, ToPILImage, Normalize
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch import nn, tensor
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from src.image_manager.super_resolution.super_resolution_data import SuperResolutionData
from utils_model import ConvolutionalBlock, ResidualBlock, SubPixelConvolutionalBlock
import shutil


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


class SRResNet(nn.Module):
    """The Super resolution ResNet model."""

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
        self.generator = Generator(large_kernel_size, small_kernel_size, n_channels, n_blocks, scaling_factor)

    def train(self, train_dataset: SuperResolutionData, val_dataset: SuperResolutionData, epochs: int,
              save_folder_model: str = "", save_folder_images: str = "", batch_size: int = 16,
              learning_rate: float = 1e-4) \
            -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
        """  #TODO implement better metric tracking (class)
        Generator can be trained before training the full GAN model to optimize convergence.
        Divide learning rate by 10 at mid training.

        Parameters
        ----------
        train_dataset: SuperResolutionData
            Dataset to use for training.
        val_dataset: SuperResolutionData
            Dataset used for validation.
        epochs: int
            Number of epochs
        save_folder_model: str, default ""
            Folder where to save the best model. If empty, the model won't be saved
        save_folder_images: str, default ""
            Folder where to save validation images (low, high and super resolution) at each validation step.
        batch_size: int, default 16
            Batch size for training
        learning_rate: float, default 1e-4
            Learning rate used for training

        Returns
        -------
        Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]
            list of training losses for each mini-batch, list of training losses for each epoch, list of metrics for
            each epoch (val set).
        """
        if save_folder_model:
            if os.path.exists(save_folder_model):
                shutil.rmtree(save_folder_model, ignore_errors=True)
            os.makedirs(save_folder_model)
        else:
            print("Model won't be saved. To save the model, please specify a save folder path.")

        # Create training process log file.
        writer = SummaryWriter(os.path.join("samples", "logs", "Generator"))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator.to(device)

        # Compute three losses to monitor the training: train loss (batch and epoch) and validation loss (epoch)
        losses_epoch = {"train": [],
                        "val": []}

        losses_batch = {"train": []}

        # Compute PSNR and SSIM on validation set for each epoch.
        metrics = {"PSNR": [],
                   "SSIM": []}

        best_val_loss = np.inf

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                                 persistent_workers=True)

        # Initialize generator's optimizer
        optimizer = Adam(params=filter(lambda p: p.requires_grad, self.generator.parameters()), lr=learning_rate)

        # Generator loss
        content_loss = MSELoss().to(device)

        start = time()

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')

            running_loss = 0.0

            self.generator.train()

            total_batch = len(data_loader)

            # lr/hr/sr: low/high/super resolution
            for i_batch, (lr_images, hr_images) in enumerate(data_loader):
                optimizer.zero_grad()

                lr_images = lr_images.to(device)  # In [0, 1] then normalized
                hr_images = hr_images.to(device)  # In [0, 1]

                sr_images = self.generator(lr_images)  # Super resolution images.

                loss = content_loss(sr_images, hr_images)  # MSE.

                # Backward step: compute gradients.
                loss.backward()

                # Step: update model parameters.
                optimizer.step()

                print(f'{i_batch + 1}/{total_batch} '
                      f'[{"=" * int(40 * (i_batch + 1) / total_batch)}>'
                      f'{"-" * int(40 - 40 * (i_batch + 1) / total_batch)}] '
                      f'- Train loss {loss.item():.4f} '
                      f'- Duration {time() - start:.1f} s\r', end="")

                # Save logs for tensorboard.
                iteration = i_batch + epoch * total_batch + 1
                writer.add_scalar("Train/generator_loss", loss.item(), iteration)

                losses_batch["train"].append(loss.item())
                running_loss += loss.item()

            losses_epoch["train"].append(running_loss / len(data_loader))

            val_loss, psnr, ssim = self.evaluate(val_dataset,
                                                 batch_size=1,  # val images can have different size.
                                                 save_folder_images=save_folder_images + "_epoch_" + str(epoch + 1))
            losses_epoch["val"].append(val_loss)
            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)

            print(f'Epoch {epoch + 1}/{epochs} '
                  f'- Train loss: {losses_epoch["train"][-1]:.4f}'
                  f'- Val Loss: {losses_epoch["val"][-1]:.4f}',
                  f'- PSNR: {psnr:.2f} '
                  f'- SSIM: {ssim:.2f} '
                  f'- Duration {time() - start:.1f} s')

            if losses_epoch["val"][-1] < best_val_loss and save_folder_model:
                best_val_loss = losses_epoch["val"][-1]
                torch.save(self.generator.state_dict(), os.path.join(save_folder_model,
                                                           f"best_generator_epoch_{epoch + 1}.torch"))

        if save_folder_model:
            torch.save(self.generator.state_dict(),
                       os.path.join(save_folder_model, f'final_generator{datetime.now().strftime("%Y%m%d%H%M")}.torch'))

        return losses_batch, losses_epoch, metrics

    def evaluate(self, val_dataset: SuperResolutionData, batch_size: int = 1, save_folder_images: str = "") \
            -> Tuple[float, float, float]:
        """
        Main function to test the model, using PSNR and SSIM.
        No validation data should be provided as GAN cannot be monitored using a validation loss.

        PSNR [dB] and SSIM measures are calculated on the y-channel of center-cropped, removal of a 4-pixel wide strip
        from each border PSNR is computed on the Y channel (luminance) of the YCbCr image.

        Parameters
        ----------
        val_dataset: SuperResolutionData
            dataset to use for testing.
        batch_size: int, default 16
            batch size for evaluation.
        save_folder_images: str, default ""
            Folder to save generated images.

        Returns
        -------
        Tuple[float, float, float]
            Average validation loss, psnr, ssim.
        """
        if save_folder_images:
            if os.path.exists(save_folder_images):
                shutil.rmtree(save_folder_images, ignore_errors=True)
            os.makedirs(save_folder_images)
        else:
            print("Images won't be saved. To save images, please specify a save folder path.")

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator.to(device)

        all_psnr = []
        all_ssim = []
        val_losses = []

        content_loss = MSELoss().to(device)

        rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)

        # reverse normalization of lr_images
        reverse_normalize = Normalize(mean=[-0.475 / 0.262, -0.434 / 0.252, -0.392 / 0.262],
                                      std=[1 / 0.262, 1 / 0.252, 1 / 0.262])  # TODO parameter
        transform = Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])

        data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,
                                 persistent_workers=True)
        total_batch = len(data_loader)

        start = time()

        with torch.no_grad():
            self.generator.eval()

            for i_batch, (lr_images, hr_images) in enumerate(data_loader):
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                sr_images = self.generator(lr_images)  # Super resolution images.

                loss = content_loss(sr_images, hr_images)  # MSE.
                val_losses.append(loss)

                # Save images.
                if save_folder_images:
                    if not os.path.exists(save_folder_images):
                        os.makedirs(save_folder_images)

                    for i in range(sr_images.size(0)):
                        images = torch.stack([transform(reverse_normalize(lr_images[i, :, :, :])),
                                              transform(sr_images[i, :, :, :]),
                                              transform(hr_images[i, :, :, :])])
                        grid = make_grid(images, nrow=3, padding=5)
                        save_image(grid,
                                   os.path.join(save_folder_images, f'image_{i_batch * batch_size + i}.png'), padding=5)

                # Compute PSNR and SSIM
                hr_images = 255 * hr_images  # Map from [0, 1] to [0, 255]
                sr_images = 255 * sr_images  # Map from [0, 1] to [0, 255]

                # Use Y channel only (luminance) to compute PSNR and SSIM (RGB to YCbCr conversion)
                sr_Y = torch.matmul(sr_images.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.
                hr_Y = torch.matmul(hr_images.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

                # Change device
                sr_Y = sr_Y.cpu().numpy()
                hr_Y = hr_Y.cpu().numpy()

                # Calculate PSNR
                batch_psnr = [peak_signal_noise_ratio(sr_Y[i], hr_Y[i], data_range=255.) for i in range(sr_Y.shape[0])]

                # Calculate SSIM
                batch_ssim = [structural_similarity(sr_Y[i], hr_Y[i], data_range=255.) for i in range(sr_Y.shape[0])]

                all_psnr.extend(batch_psnr)
                all_ssim.extend(batch_ssim)

                print(f'{i_batch + 1}/{total_batch} '
                      f'[{"=" * int(40 * (i_batch + 1) / total_batch)}>'
                      f'{"-" * int(40 - 40 * (i_batch + 1) / total_batch)}] '
                      f'- Duration {time() - start:.1f} s\r', end="")

        return sum(val_losses) / len(val_losses), sum(all_psnr) / len(all_psnr), sum(all_ssim) / len(all_ssim)

    def load(self, generator: Union[Generator, str]):
        """
        Load a pretrained model.

        Parameters
        ----------
        generator: Union[Generator, str]
            A path to a pretrained model, or a torch model.

        Raises
        ------
        TypeError
            If 'generator' type is not a string or a model.

        """
        if isinstance(generator, str):
            self.generator.load_state_dict(torch.load(generator))  # From path
        elif isinstance(generator, type(self.generator)):
            self.generator.load_state_dict(generator.state_dict())
        else:
            raise TypeError("Generator argument must be either a path to a trained model, or a trained model.")
