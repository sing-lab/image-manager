"""SrResNet_training_V1 model."""
from math import ceil
import os
import shutil
import numpy as np
from time import time
from typing import Tuple, Union, Optional
from datetime import datetime

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, ToPILImage
from models.super_resolution_model_base import SuperResolutionModelBase
from models.SRGAN.generator import Generator
from models.utils_models import get_tiles_from_image, get_image_from_tiles, RGB_WEIGHTS

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from super_resolution_data import SuperResolutionData


class SRResNet(SuperResolutionModelBase):
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
              experiment_name: str, model_save_folder: str = "", images_save_folder: str = "", batch_size: int = 16,
              learning_rate: float = 1e-4) -> None:
        """
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
        experiment_name: str
            Name of the experiment.
        model_save_folder: str, default ""
            Folder where to save the best model. If empty, the model won't be saved
        images_save_folder: str, default ""
            Folder where to save validation images (low, high and super resolution) at each validation step.
        batch_size: int, default 16
            Batch size for training
        learning_rate: float, default 1e-4
            Learning rate used for training

        """
        if model_save_folder:
            if os.path.exists(model_save_folder):
                shutil.rmtree(model_save_folder, ignore_errors=True)
            os.makedirs(model_save_folder)
        else:
            print("Model won't be saved. To save the model, please specify a save folder path.")

        # Create training process log file.
        writer = SummaryWriter(os.path.join("samples", "logs", experiment_name))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator.to(device)

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                                 persistent_workers=True)

        best_val_loss = np.inf

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
            # Training step.
            for i_batch, (lr_images, hr_images) in enumerate(data_loader):

                lr_images = lr_images.to(device)  # In [0, 1] then normalized
                hr_images = hr_images.to(device)  # In [0, 1]

                sr_images = self.generator(lr_images)  # Super resolution images.

                loss = content_loss(sr_images, hr_images)  # MSE.

                # Backward step: compute gradients.
                loss.backward()

                # Step: update model parameters.
                optimizer.step()
                self.generator.zero_grad(set_to_none=True)

                print(f'{i_batch + 1}/{total_batch} '
                      f'[{"=" * int(40 * (i_batch + 1) / total_batch)}>'
                      f'{"-" * int(40 - 40 * (i_batch + 1) / total_batch)}] '
                      f'- Train loss {loss.item():.4f} '
                      f'- Duration {time() - start:.1f} s\r', end="")

                # Save logs for tensorboard.
                iteration = i_batch + epoch * total_batch + 1
                writer.add_scalar("Train/generator_loss", loss.item(), iteration)

                running_loss += loss.item()

            # Evaluation step.
            val_loss, psnr, ssim = self.evaluate(val_dataset,
                                                 batch_size=1,  # val images can have different size.
                                                 images_save_folder=f"{images_save_folder}_epoch_{epoch + 1}")

            writer.add_scalar("Val/PSNR", psnr, epoch + 1)
            writer.add_scalar("Val/SSIM", ssim, epoch + 1)
            writer.add_scalar("Val/Loss", val_loss, epoch + 1)

            print(f'Epoch {epoch + 1}/{epochs} '
                  f'- Train loss: {running_loss / len(data_loader):.4f}'
                  f'- Val Loss: {val_loss:.4f}',
                  f'- PSNR: {psnr:.2f} '
                  f'- SSIM: {ssim:.2f} '
                  f'- Duration {time() - start:.1f} s')

            if val_loss < best_val_loss and model_save_folder:
                best_val_loss = val_loss
                torch.save(self.generator.state_dict(), os.path.join(model_save_folder,
                                                                     f"best_generator_epoch_{epoch + 1}.torch"))

        if model_save_folder:
            torch.save(self.generator.state_dict(),
                       os.path.join(model_save_folder, f'final_generator{datetime.now().strftime("%Y%m%d%H%M")}.torch'))

        return

    def evaluate(self, val_dataset: SuperResolutionData, batch_size: int = 1, images_save_folder: str = "",
                 reverse_normalize: bool = True) -> Tuple[float, float, float]:
        """
        Main function to test the model, using PSNR and SSIM.

        PSNR [dB] and SSIM measures are calculated on the y-channel of center-cropped, removal of a 4-pixel wide strip
        from each border PSNR is computed on the Y channel (luminance) of the YCbCr image.

        Parameters
        ----------
        val_dataset: SuperResolutionData
            dataset to use for testing.
        batch_size: int, default 16
            batch size for evaluation.
        images_save_folder: str, default ""
            Folder to save generated images.
        reverse_normalize: bool, default True
            Whether to reverse image normalization before saving images or not.

        Returns
        -------
        Tuple[float, float, float]
            Average validation loss, PSNR and SSIM.
        """
        if images_save_folder:
            if os.path.exists(images_save_folder):
                shutil.rmtree(images_save_folder, ignore_errors=True)
            os.makedirs(images_save_folder)
        else:
            print("Images won't be saved. To save images, please specify a save folder path.")

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator.to(device)

        all_psnr = []
        all_ssim = []
        val_losses = []

        content_loss = MSELoss().to(device)

        transform = Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])

        data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,
                                 persistent_workers=True)
        total_batch = len(data_loader)

        start = time()

        self.generator.eval()

        with torch.no_grad():

            for i_batch, (lr_images, hr_images) in enumerate(data_loader):
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                sr_images = self.generator(lr_images)  # Super resolution images.

                loss = content_loss(sr_images, hr_images)  # MSE.
                val_losses.append(loss)

                # Save images.
                if images_save_folder and (total_batch <= 10 or i_batch % (total_batch // 10) == 0):
                    for i in range(sr_images.size(0)):
                        if reverse_normalize:
                            # Reverse normalization of lr_images
                            lr_images[i, :, :, :] = val_dataset.reverse_normalize(lr_images[i, :, :, :])

                        images = torch.stack([transform(lr_images[i, :, :, :]),
                                              transform(sr_images[i, :, :, :]),
                                              transform(hr_images[i, :, :, :])])
                        grid = make_grid(images, nrow=3, padding=5)
                        save_image(grid,
                                   os.path.join(images_save_folder, f'image_{i_batch * batch_size + i}.png'), padding=5)

                # Compute PSNR and SSIM
                hr_images = 255 * hr_images  # Map from [0, 1] to [0, 255]
                sr_images = 255 * sr_images  # Map from [0, 1] to [0, 255]

                # Use Y channel only (luminance) to compute PSNR and SSIM (RGB to YCbCr conversion)
                sr_Y = torch.matmul(sr_images.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], RGB_WEIGHTS.to(device)) / 255. + 16.
                hr_Y = torch.matmul(hr_images.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], RGB_WEIGHTS.to(device)) / 255. + 16.

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

    def predict(self, test_dataset: SuperResolutionData, images_save_folder: str = "", batch_size: int = 1,
                force_cpu: bool = True, tile_size: Optional[int] = None, tile_overlap: Optional[int] = None,
                tile_batch_size: Optional[int] = None, scaling_factor: int = 4) -> None:
        """
        Process an image into super resolution.
        We use high resolution images as input, therefore test_dataset should have parameter normalize_hr set to True.

        Parameters
        ----------
        test_dataset: SuperResolutionData
            The images to process.
        batch_size: int, default 1
            The batch size for predictions. If prediction made by tiles, batch_size should be 1.
        tile_batch_size: Optional[int], default None
            Images are processed one by one, however tiles for a given image can be processed by batches.
        images_save_folder: str
            The folder where to save predicted images.
        force_cpu: bool
            Whether to force usage of CPU or not (inference on high resolution images may run GPU out of memory).
        tile_size: Optional[int], default None
            As too large images result in the out of GPU memory issue, tile option will first crop input images into
            tiles, then process each of them. Finally, they will be merged into one image. 0: not used tiles.
            It is advised to use the same tile_size as low resolution image in the training.
            Adapted from https://github.com/ata4/esrgan-launcher/blob/master/upscale.py
        tile_overlap: Optional[int], default None
            Overlap pixels between tiles.
        scaling_factor: int, default 4
            The scaling factor to use when downscaling high resolution images into low resolution images.

        Raises
        ------
        ValueError
            If 'tile_size' is not None and 'batch_size' is not 1, as prediction by tiles only supports batch_size of 1.
        """
        if images_save_folder:
            if os.path.exists(images_save_folder):
                shutil.rmtree(images_save_folder, ignore_errors=True)
            os.makedirs(images_save_folder)
        else:
            print("Images won't be saved. To save images, please specify a save folder path.")

        if bool(test_dataset.normalize_hr) == bool(tile_size):
            raise ValueError("When using prediction by tile, normalize_hr should be False as we normalize each tile"
                             "one by one. When using normal prediction, normalize_hr should be True as we normalize the "
                             "full image")

        if tile_size and batch_size != 1:
            raise ValueError(f"Prediction is made by tile as 'tile_size' is specified. Only batch_size of 1"
                             f" is supported, but is '{batch_size}'. To predict tiles by batch, please use "
                             f"'tile_batch_size'.")

        device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
        self.generator.to(device)
        self.generator.eval()

        # pin_memory can lead to too much pagination memory needed.
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        total_batch = len(data_loader)

        start = time()

        with torch.no_grad():
            for i_batch, (_, images) in enumerate(data_loader):

                if tile_size:
                    # Normalization will be applied on each tile independently.
                    image = images[0]  # batch_size must be 1

                    # 1. Get tiles from input image
                    tiles = get_tiles_from_image(image, tile_size, tile_overlap)

                    # Retrieve tiles by row and by col to merge predictions
                    channel, height, width = image.shape
                    tiles_x = ceil(width / tile_size)
                    tiles_y = ceil(height / tile_size)

                    sr_tiles = torch.empty(
                        (tiles_x * tiles_y, channel, scaling_factor * (tile_size + 2 * tile_overlap),
                         scaling_factor * (tile_size + 2 * tile_overlap)))

                    # 2. Loop over all tiles to make predictions

                    batches = torch.split(tiles, tile_batch_size, dim=0)  # Create batches of tiles
                    total_batch_tile = len(batches)

                    for i_batch_tile, batch in enumerate(batches):
                        print(f'{i_batch_tile + 1}/{total_batch_tile} '
                              f'[{"=" * int(40 * (i_batch_tile + 1) / total_batch_tile)}>'
                              f'{"-" * int(40 - 40 * (i_batch_tile + 1) / total_batch_tile)}] '
                              f'- Duration {time() - start:.1f} s\r', end="")
                        index_start = i_batch_tile * tile_batch_size
                        index_end = min((i_batch_tile + 1) * tile_batch_size, tiles.size()[0])  # Last batch may be smaller.
                        batch = test_dataset.normalize(batch)  # Normalize each tile.
                        sr_tiles[index_start: index_end] = self.generator(batch.to(device))

                    # 3. Merge upscaled tiles: retrieve index and position from indexes
                    sr_images = get_image_from_tiles(sr_tiles, tile_size, tile_overlap, scaling_factor, image)

                else:
                    sr_images = self.generator(images.to(device))

                # Save images
                if images_save_folder:
                    for i in range(sr_images.size(0)):
                        save_image(sr_images[i, :, :, :],
                                   os.path.join(images_save_folder, f'{i_batch + i}.png'))  #TODO change extension with original extension

                print(f'{i_batch + 1}/{total_batch} '
                      f'[{"=" * int(40 * (i_batch + 1) / total_batch)}>'
                      f'{"-" * int(40 - 40 * (i_batch + 1) / total_batch)}] '
                      f'- Duration {time() - start:.1f} s\r', end="")

        return

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
