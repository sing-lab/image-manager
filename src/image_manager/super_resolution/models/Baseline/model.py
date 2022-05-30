"""Bicubic interpolation (baseline)."""
#TODO: only one line of code is needed?
import os
import shutil
from time import time
from typing import Tuple

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, ToPILImage, Normalize, InterpolationMode
from torchvision.utils import save_image, make_grid

from src.image_manager.super_resolution.super_resolution_data import SuperResolutionData


class BicubicInterpolation:
    """Simple class to perform bicubic up-sampling"""

    def __init__(self):
        pass

    def train(self):
        pass

    def evaluate(self, val_dataset: SuperResolutionData, batch_size: int = 16, images_save_folder: str = "") \
            -> Tuple[float, float]:
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

        all_psnr = []
        all_ssim = []

        #TODO chek if dataset is norm and param reverse_norm is false

        rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)

        transform = Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])

        data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,
                                 persistent_workers=True)
        total_batch = len(data_loader)

        start = time()

        with torch.no_grad():

            for i_batch, (lr_images, hr_images) in enumerate(data_loader):

                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                high_res_height, high_res_width = hr_images.size()[1:]
                sr_images = Resize((high_res_height, high_res_width), interpolation=InterpolationMode.BICUBIC)(lr_images)

                # Save images.
                if images_save_folder and (total_batch <= 10 or i_batch % (total_batch // 10) == 0):
                    for i in range(sr_images.size(0)):
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


    def predict(self):
        # bicubic =
        pass

    def load(self):
        pass

