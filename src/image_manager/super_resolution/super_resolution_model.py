"""Class for super resolution model.""" #TODO useful? alternative: in main.py: config file has the enum name inside
from models.model_enum import ModelEnum, get_model_from_enum
from src.image_manager.super_resolution.models.super_resolution_model_base import SuperResolutionModelBase
from utils import load_config


class SuperResolutionModel(SuperResolutionModelBase):
    """Class for super resolution model."""

    def __init__(self, model_name: ModelEnum = ModelEnum.SRGAN, from_pretrained: bool = True):
        self.model = get_model_from_enum(model_name, from_pretrained)

    def train(self, train_dataset, val_dataset, **kwargs):  #TODO other parameter in config file
        self.model.train(train_dataset, val_dataset, **kwargs)

    def evaluate(self, val_dataset, **kwargs):
        self.model.evaluate(val_dataset, **kwargs)
        pass

    def predict(self, val_dataset, **kwargs):
        """

        Returns
        -------

        """
        self.model.predict(val_dataset, **kwargs)
        pass

    def load(self):
        pass