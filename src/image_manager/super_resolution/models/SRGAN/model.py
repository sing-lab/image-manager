"""SRGAN model implementation"""
import os
import shutil
from time import time
from typing import Tuple, Optional, Union

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, ToPILImage
from torchvision.transforms import Normalize
from torchvision.utils import save_image, make_grid

from models.SRGAN.discriminator import Discriminator
from models.SRGAN.generator import Generator
from super_resolution_data import SuperResolutionData
from models.super_resolution_model_base import SuperResolutionModelBase
from models.SRGAN.utils_loss import TruncatedVGG


class SRGAN(SuperResolutionModelBase):
    """Super resolution with GAN model"""

    def __init__(self, discriminator: Discriminator, generator: Generator, truncated_vgg: TruncatedVGG):
        """

        Parameters
        ----------
        discriminator: Discriminator
            The discriminator model. Classify images as real or fake.
        generator: Generator
            The generator model. Generates fake images.
        truncated_vgg: TruncatedVGG19
            The truncated VGG network used to compute the perceptual loss.
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.truncated_vgg = truncated_vgg

    def train(self, train_dataset: SuperResolutionData, val_dataset: SuperResolutionData, epochs: int,
              experiment_name: str, model_save_folder: str = "", images_save_folder: str = "", batch_size: int = 16,
              learning_rate: float = 1e-4, beta_loss: float = 1e-3) -> None:
        """
        Main function to train the model and save the final model.
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
        beta_loss: float, default 1e-3
            The coefficient to weight the adversarial loss in the perceptual loss.

        """
        if model_save_folder:
            if os.path.exists(model_save_folder):
                shutil.rmtree(model_save_folder, ignore_errors=True)
            os.makedirs(model_save_folder)
        else:
            print("Model won't be saved. To save the model, please specify a save folder path.")

        # Create log file to monitore training and evaluation.
        writer = SummaryWriter(os.path.join("samples", "logs", experiment_name))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator.to(device)
        self.discriminator.to(device)

        self.truncated_vgg.to(device)
        self.truncated_vgg.eval()  # Used to compute the content loss only.

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                                 persistent_workers=True)

        # Initialize generator's optimizer
        optimizer_g = Adam(params=filter(lambda p: p.requires_grad, self.generator.parameters()), lr=learning_rate)
        scheduler_g = MultiStepLR(optimizer_g, milestones=[epochs // 2], gamma=0.1)

        # Initialize discriminator's optimizer
        optimizer_d = Adam(params=filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=learning_rate)
        scheduler_d = MultiStepLR(optimizer_d, milestones=[epochs // 2], gamma=0.1)

        # Generator loss
        content_loss = MSELoss().to(device)

        # Discriminator loss
        adversarial_loss = BCEWithLogitsLoss().to(device)

        start = time()

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')

            running_adversarial_loss_d = 0.0
            running_adversarial_loss_g = 0.0
            running_content_loss_g = 0.0

            self.generator.train()
            self.discriminator.train()

            total_batch = len(data_loader)

            # lr/hr/sr: low/high/super resolution
            # Training step.
            for i_batch, (lr_images, hr_images) in enumerate(data_loader):
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                sr_images = self.generator(lr_images)  # Super resolution images in [0, 1].

                # GENERATOR
                # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
                sr_images_vgg = self.truncated_vgg(sr_images)
                hr_images_vgg = self.truncated_vgg(hr_images).detach()  # Detach as they don't need gradient (targets).

                # Images discrimination
                sr_discriminated = self.discriminator(sr_images)  # (N)

                # Calculate the Perceptual loss
                # MSE on VGG space + MSE on real space.  MSE on raw images != MSE on normalized images.
                content_loss_g = content_loss(sr_images_vgg, hr_images_vgg)
                adversarial_loss_g = adversarial_loss(sr_discriminated, torch.ones_like(sr_discriminated))
                perceptual_loss_g = content_loss_g + beta_loss * adversarial_loss_g

                # Backward step: compute gradients.
                perceptual_loss_g.backward()

                # Step: update model parameters.
                optimizer_g.step()
                self.generator.zero_grad(set_to_none=True)

                # DISCRIMINATOR
                hr_discriminated = self.discriminator(hr_images)
                sr_discriminated = self.discriminator(sr_images.detach().clone())
                # Don't use previous sr_discriminated because it would also update generator parameters.

                d_sr_probability = torch.sigmoid_(torch.mean(sr_discriminated.detach()))  # prob of sr
                d_hr_probability = torch.sigmoid_(torch.mean(hr_discriminated.detach()))  # prob of hr

                # # Binary Cross-Entropy loss
                adversarial_loss_d = adversarial_loss(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                                     adversarial_loss(hr_discriminated, torch.ones_like(hr_discriminated))

                # Backward step: compute gradients.
                adversarial_loss_d.backward()

                # Step: update model parameters.
                optimizer_d.step()
                self.discriminator.zero_grad(set_to_none=True)

                print(f'{i_batch + 1}/{total_batch} '
                      f'[{"=" * int(40 * (i_batch + 1) / total_batch)}>'
                      f'{"-" * int(40 - 40 * (i_batch + 1) / total_batch)}] '
                      f'- Loss generator (adversarial) {adversarial_loss_g.item():.4f} '
                      f'- Loss generator (content) {content_loss_g.item():.4f} '
                      f'- Loss discriminator (adversarial) {adversarial_loss_d.item():.4f} '
                      f'- Duration {time() - start:.1f} s\r', end="")

                # Save logs for tensorboard.
                iteration = (i_batch + epoch * total_batch + 1)
                writer.add_scalar("Train/discriminator_total_loss", adversarial_loss_d.item(), iteration)
                writer.add_scalar("Train/generator_content_loss", content_loss_g.item(), iteration)
                writer.add_scalar("Train/generator_adversarial_loss", adversarial_loss_g.item(), iteration)
                writer.add_scalar("Train/generator_total_loss", content_loss_g.item() + adversarial_loss_g.item(),
                                  iteration)
                writer.add_scalar("Train/discriminator_hr_probability", d_hr_probability, iteration)
                writer.add_scalar("Train/discriminator_sr_probability", d_sr_probability, iteration)

                running_adversarial_loss_d += adversarial_loss_d.item()
                running_adversarial_loss_g += adversarial_loss_g.item()
                running_content_loss_g += content_loss_g.item()

            scheduler_g.step()
            scheduler_d.step()

            # Evaluation step.
            psnr, ssim = self.evaluate(val_dataset,
                                       batch_size=1,
                                       images_save_folder="{images_save_folder}_epoch_{str(epoch + 1)}")

            writer.add_scalar("Val/PSNR", psnr, epoch + 1)
            writer.add_scalar("Val/SSIM", ssim, epoch + 1)

            print(f'Epoch {epoch + 1}/{epochs} '
                  f'- Loss generator (adversarial): {running_adversarial_loss_g / len(data_loader):.4f} '
                  f'- Loss discriminator (adversarial): {running_adversarial_loss_d / len(data_loader):.4f} '
                  f'- Loss generator (content): {running_content_loss_g / len(data_loader):.4f} '
                  f'- PSNR: {psnr:.2f} '
                  f'- SSIM: {ssim:.2f} '
                  f'- Duration {time() - start:.1f} s')

            if model_save_folder:
                torch.save(self.discriminator.state_dict(),
                           os.path.join(model_save_folder, f"discriminator_epoch_{epoch + 1}.torch"))
                torch.save(self.generator.state_dict(),
                           os.path.join(model_save_folder, f"generator_epoch_{epoch + 1}.torch"))

            # Free some memory since their histories may be stored
            del lr_images, hr_images, sr_images, hr_images_vgg, sr_images_vgg, hr_discriminated, sr_discriminated

        return

    def evaluate(self, val_dataset: SuperResolutionData, batch_size: int = 1, images_save_folder: str = "",
                 reverse_normalize: bool = True) -> Tuple[float, float]:
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
        images_save_folder: str, default ""
            Folder to save generated images.
        reverse_normalize: bool, default True
            Whether to reverse image normalization before saving images or not.

        Returns
        -------
        Tuple[float, float]
            Average PSNR and SSIM values.
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

        rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)

        # Reverse normalization of lr_images
        if reverse_normalize:
            denormalize = Normalize(mean=[-0.475 / 0.262, -0.434 / 0.252, -0.392 / 0.262],
                                    std=[1 / 0.262, 1 / 0.252, 1 / 0.262])

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

                # Save images.
                if images_save_folder and (total_batch <= 10 or i_batch % (total_batch // 10) == 0):
                    for i in range(sr_images.size(0)):
                        if reverse_normalize:
                            lr_images[i, :, :, :] = denormalize(lr_images[i, :, :, :])

                        images = torch.stack([transform(lr_images[i, :, :, :]),
                                              transform(sr_images[i, :, :, :]),
                                              transform(hr_images[i, :, :, :])])
                        grid = make_grid(images, nrow=3, padding=5)
                        save_image(grid,
                                   os.path.join(images_save_folder, f'image_{i_batch * batch_size + i}.png'), padding=5)

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

        return sum(all_psnr) / len(all_psnr), sum(all_ssim) / len(all_ssim)

    def predict(self, test_dataset: SuperResolutionData, batch_size: int = 1, images_save_folder: str = "",
                force_cpu: bool = True) -> None:
        """
        Process an image into super resolution.
        We use high resolution images as input, therefore test_dataset should have parameter normalize_hr set to True.

        Parameters
        ----------
        test_dataset: SuperResolutionData
            The images to process.
        batch_size: int, default 1
            The batch size for predictions.
        images_save_folder: str
            The folder where to save predicted images.
        force_cpu: bool
            Whether to force usage of CPU or not (inference on high resolution images may run GPU out of memory).

        """
        if images_save_folder:
            if os.path.exists(images_save_folder):
                shutil.rmtree(images_save_folder, ignore_errors=True)
            os.makedirs(images_save_folder)
        else:
            print("Images won't be saved. To save images, please specify a save folder path.")

        if not test_dataset.normalize_hr:
            print("Warning! As we use high resolution images as model input, test_dataset should have 'normalize_hr' "
                  "set to True (currently False).")

        device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
        self.generator.to(device)
        self.generator.eval()

        # pin_memory can lead to too much pagination memory needed.
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        total_batch = len(data_loader)

        start = time()

        with torch.no_grad():
            for i_batch, (lr_images, hr_images) in enumerate(data_loader):
                hr_images = hr_images.to(device)
                sr_images = self.generator(hr_images)

                # Save images
                if images_save_folder:
                    for i in range(sr_images.size(0)):
                        save_image(sr_images[i, :, :, :],
                                   os.path.join(images_save_folder, f'{i_batch * batch_size + i}.png'))

                print(f'{i_batch + 1}/{total_batch} '
                      f'[{"=" * int(40 * (i_batch + 1) / total_batch)}>'
                      f'{"-" * int(40 - 40 * (i_batch + 1) / total_batch)}] '
                      f'- Duration {time() - start:.1f} s\r', end="")

        return

    def load(self, generator: Optional[Union[Generator, str]] = None,
             discriminator: Optional[Union[Discriminator, str]] = None):
        """
        Load a pretrained model

        Parameters
        ----------
        generator: Optional[Union[Generator, str]], default None
            A path to a pretrained model, or a torch model.
        discriminator: Optional[Union[Discriminator, str]]: default None
            A path to a pretrained model, or a torch model.

        Raises
        ------
        TypeError
            If 'generator' or 'discriminator' type is not a string or a model.

        """
        if generator is not None:
            if isinstance(generator, str):
                self.generator.load_state_dict(torch.load(generator))  # From path
            elif isinstance(generator, type(self.generator)):
                self.generator.load_state_dict(generator.state_dict())
            else:
                raise TypeError("Generator argument must be either a path to a trained model, or a trained model.")

        if discriminator is not None:
            if isinstance(discriminator, str):
                self.discriminator.load_state_dict(torch.load(discriminator))  # From path
            elif isinstance(discriminator, Discriminator):
                self.discriminator.load_state_dict(discriminator.state_dict())
            else:
                raise TypeError("Generator argument must be either a path to a trained model, or a trained model.")
