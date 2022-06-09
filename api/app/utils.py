from models.model_enum import ModelEnum, get_model_from_enum
from super_resolution_data import SuperResolutionData
import tempfile
from PIL import Image
import io
import os

# Loading model.
model_enum = ModelEnum["SRGAN"]
model = get_model_from_enum(model_enum, from_pretrained=True)


def get_prediction_from_bytes(image_bytes: bytes) -> bytes:
    """

    Parameters
    ----------
    image_bytes

    Returns
    -------

    """