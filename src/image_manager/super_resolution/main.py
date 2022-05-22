"""Main module to train the model."""
import os
import sys

# TODO seed + namespace imports
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # Better performances.

sys.path.append(
    "C:\\Users\\Mathias\\Documents\\Projets_Python\\image_manager\\src\\image_manager\\super_resolution\\models\\SRGAN")
sys.path.append(
    "C:\\Users\\Mathias\\Documents\\Projets_Python\\image_manager\\src\\image_manager\\super_resolution\\models\\SRResNet")
from models.model_enum import ModelEnum, get_model_from_enum

from src.image_manager.super_resolution.super_resolution_data import SuperResolutionData
from src.image_manager.super_resolution.utils import load_config

if __name__ == "__main__":

    # Loading config.
    config_path = "../../../configs/SRGAN/model_config.yml"  # TODO use arg() or parameter
    config = load_config(config_path)

    # Loading model.
    model_enum = ModelEnum[config["model_type"]]
    model = get_model_from_enum(model_enum, from_pretrained=bool(config["from_pretrained"]))

    # Running task
    if config["task"] == "train":

        # Loading data.
        val_dataset = SuperResolutionData(image_folder=config["paths"]["val_images_input_path"],
                                          is_train_split=False,
                                          normalize_lr=True)

        train_dataset = SuperResolutionData(image_folder=config["paths"]["train_images_input_path"],
                                            is_train_split=True,
                                            normalize_lr=True)
        # TODO **config with train with **kwargs. (section train_parameters a ajouter puis **config[trained_parm]
        # une config par type de model.
        model.train(train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    model_save_folder=os.path.join(config["paths"]['model_save_folder'], config['experiment_name']),
                    images_save_folder=os.path.join(config["paths"]["val_images_save_folder"], config["experiment_name"]),
                    experiment_name=config["experiment_name"],
                    **config["hyperparameters"]
                    )

    if config["task"] == "evaluate":
        pass

    if config["task"] == 'predict':
        pass

