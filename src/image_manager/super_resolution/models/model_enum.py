"""List of trained models."""
from enum import Enum
from models.SRResNet.model import SRResNet

from models.SRGAN.discriminator import Discriminator
from models.SRGAN.generator import Generator
from models.SRGAN.model import SRGAN
from models.SRGAN.utils_loss import TruncatedVGG

from models.super_resolution_model_base import SuperResolutionModelBase


class ModelEnum(Enum):
    """Enum with models that we can use for super resolution."""
    SRGAN = 0,
    SRRESNET = 1,
    ESRGAN = 2,
    BASELINE = 3,


def get_model_from_enum(model: ModelEnum, from_pretrained: bool = False) -> SuperResolutionModelBase:
    """
    Get specific model from enum.  #TODO change path for app.

    SRGAN: never trained from scratched.
        - Loaded by default with pretrained generator.
        If from_pretrained is True, will be loaded with pretrained generator and discriminator (fully trained).

    Parameters
    ----------
    model: ModelEnum
    from_pretrained: bool, default False
        Whether to load a pretrained model or  the raw model.

    Returns
    -------
    SuperResolutionModel
        model from enum
    """
    if model.name.lower() == 'srgan':
        base_model = SRGAN(discriminator=Discriminator(), generator=Generator(), truncated_vgg=TruncatedVGG())
        if from_pretrained:
            try:
                base_model.load(generator="../../../models/super_resolution/SRGAN/Article/generator.torch",
                                discriminator="../../../models/super_resolution/SRGAN/Article/discriminator.torch")
            except FileNotFoundError:
                base_model.load(generator="../../models/super_resolution/SRGAN/Article/generator.torch",
                                discriminator="../../models/super_resolution/SRGAN/Article/discriminator.torch")
        else:
            try:
                base_model.load(
                    generator="../../../models/super_resolution/SrResNet/training_V1/best_generator_epoch_30.torch")
            except FileNotFoundError:
                base_model.load(
                    generator="../../models/super_resolution/SrResNet/training_V1/best_generator_epoch_30.torch")
        return base_model

    if model.name.lower() == 'srresnet':
        base_model = SRResNet()
        if from_pretrained:
            try:
                base_model.load('../../../models/super_resolution/SrResNet/training_V1/best_generator_epoch_30.torch')
            except FileNotFoundError:
                base_model.load('../../models/super_resolution/SrResNet/training_V1/best_generator_epoch_30.torch')

        return base_model

    raise Exception('Model name not defined in enum')
