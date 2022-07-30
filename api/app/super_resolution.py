"""Module for super resolution in the app"""
import os
from tempfile import TemporaryDirectory

from PIL import Image

from super_resolution_data import SuperResolutionData
from models.model_enum import ModelEnum, get_model_from_enum

Image.MAX_IMAGE_PIXELS = None  # Avoid Pillow DecompressionBomb>error when opening too big images

model_enum = ModelEnum["SRGAN"]
model = get_model_from_enum(model_enum, from_pretrained=False)  # Only trained generator not discriminator.


def get_prediction(image: Image) -> Image:
    """
    Process an image.
    """
    images_save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
    with TemporaryDirectory() as dataset_folder:
        image.save(os.path.join(dataset_folder, image.filename))
        dataset = SuperResolutionData(image_folder=dataset_folder, scaling_factor=4, crop_type='no_crop',
                                      normalize_hr=False)
        try:
            print("Using GPU", flush=True)
            model.predict(test_dataset=dataset, images_save_folder=images_save_folder,
                          force_cpu=False, tile_batch_size=4, tile_size=96, tile_overlap=10)
        except (RuntimeError, OSError):  # CUDA out of memory: try to predict using CPU only.
            print("Not enough GPU memory: will run on CPU.", flush=True)
            model.predict(test_dataset=dataset, images_save_folder=images_save_folder,
                          force_cpu=True, tile_batch_size=8, tile_size=96, tile_overlap=10)

    return Image.open(os.path.join(images_save_folder, "0.png"))
