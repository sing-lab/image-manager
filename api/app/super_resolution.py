import os
from shutil import copy2
from tempfile import TemporaryDirectory
from PIL import Image

from super_resolution_data import SuperResolutionData
from models.model_enum import ModelEnum, get_model_from_enum

model_enum = ModelEnum["SRGAN"]
model = get_model_from_enum(model_enum, from_pretrained=True)


def get_prediction(image_path):
    """
    Process an image.
    """
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    with TemporaryDirectory() as dataset_folder:
        image.save(os.path.join(dataset_folder, image_name))
        dataset = SuperResolutionData(image_folder=dataset_folder, is_train_split=False, scaling_factor=4,
                                      normalize_hr=True)

        try:
            model.predict(test_dataset=dataset, images_save_folder=os.path.join(dataset_folder, "predictions"),
                          force_cpu=False)
        except RuntimeError:  # CUDA out of memory: try to predict using CPU only.
            model.predict(test_dataset=dataset, images_save_folder=os.path.join(dataset_folder, "predictions"),
                          force_cpu=True)

        # Save image for user display.
        sr_image_path = os.path.join("static", "images", f"sr_{image_name}")
        copy2(os.path.join(dataset_folder, "predictions", "0.png"), sr_image_path)

    return sr_image_path
