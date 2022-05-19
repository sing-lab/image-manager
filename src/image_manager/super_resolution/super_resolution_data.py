"""Module defining class to store data or super resolution model."""
from typing import Tuple

from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, CenterCrop, Resize, InterpolationMode, Normalize, \
    ToPILImage
import os
from PIL import Image


class SuperResolutionData(Dataset):
    """
    Class for data loader
    Each High Resolution image is cropped to a squared size, then downscaled to make the Low resolution image.
    """

    def __init__(self, image_folder: str, is_train_split: bool = False, crop_size: int = 96, scaling_factor: int = 4,
                 normalize_lr: bool = True, normalize_hr: bool = False):
        """

        Parameters
        ----------
        image_folder: str
            A folder containing images.
        crop_size: int, default 96
            The target size for high resolution images for train set. Test set: images are not cropped to a fixed size.
        scaling_factor: int, default 4
            The scaling factor to use when downscaling high resolution images into low resolution images.
        normalize_lr: bool, default True
            Whether to normalize images (using imageNET statistics as we use similar images) or not.
        normalize_hr: bool, default False
            Whether to normalize images (using imageNET statistics as we use similar images) or not.
        is_train_split: bool, default False
            Whether the current split it a train split (test and validation data should not use RandomCrop).

        Raises
        ------
        ValueError
            If the crop_size is not divisible by scaling_factor, or split not in 'train', 'test', 'validation'.

        """
        if crop_size is not None and crop_size % scaling_factor != 0:
            raise ValueError("Crop size is not divisible by scaling factor! This will lead to a mismatch in the \
                              dimensions of the original high resolution patches and their super-resolved \
                              (reconstructed) versions!")

        self.images_path = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
        self.image_folder = image_folder
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.is_train_split = is_train_split
        self.normalize_lr = normalize_lr
        self.normalize_hr = normalize_hr

        # Normalize images using ImageNET dataset statistics.
        self.normalize = Normalize(mean=[0.475, 0.434, 0.392], std=[0.262, 0.252, 0.262])

    def __getitem__(self, idx: int) -> Tuple[tensor, tensor]:
        """
        Utility function.

        Parameters
        ----------
        idx: int
            index of the element to get

        Returns
        -------
        Tuple[tensor, tensor]
            The low resolution image and the high resolution image
        """
        image = Image.open(self.images_path[idx])
        image = image.convert('RGB')

        # 1. Crop original image to make the high resolution image (in [0, 1]).
        if self.is_train_split:
            transform_hr = Compose([RandomCrop(self.crop_size), ToTensor()])
        else:
            # Take the largest possible (squared) center-crop such that dimensions are divisible by the scaling factor.
            original_height, original_width = image.size
            crop_size = min((original_height - original_height % self.scaling_factor,
                             original_width - original_width % self.scaling_factor))
            transform_hr = Compose([CenterCrop(crop_size), ToTensor()])

        # 2. Downscale image to  make the low resolution image.
        high_res_image = transform_hr(image)
        high_res_height, high_res_width = high_res_image.shape[1:]
        
        transform_lr = Compose([ToPILImage(),  # Needed as interpolation can scale pixels out of [0, 1] (overshoot).
                                Resize((high_res_height // self.scaling_factor, high_res_width // self.scaling_factor),
                                interpolation=InterpolationMode.BICUBIC),
                                ToTensor()
                                ])

        low_res_image = transform_lr(high_res_image)

        if self.normalize_lr:
            low_res_image = self.normalize(low_res_image)

        if self.normalize_hr:
            high_res_image = self.normalize(high_res_image)

        return low_res_image, high_res_image

    def __len__(self) -> int:
        """
        Len function

        Returns
        -------
        int
            Number of images in the dataset.
        """
        return len(self.images_path)
