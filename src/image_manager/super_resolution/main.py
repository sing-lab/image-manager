"""Main module to train / test a model or make predictions."""
import sys

# Add thr project root in path to be able to use absolute import from this root on other submoduels.
# For docker image use "Export"
sys.path.append("C:\\Users\\Mathias\\Documents\\Projets_Python\\image_manager\\src\\image_manager\\super_resolution")
import os
from time import time

from models.model_enum import ModelEnum, get_model_from_enum
from super_resolution_data import SuperResolutionData
from utils import load_config

# TODO seed.
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # Better performances.


if __name__ == "__main__":
    start = time()

    # Loading config.
    config_path = "../../../configs/SRGAN/srgan_predict_config.yml"  # TODO use arg() or parameter
    config = load_config(config_path)

    # Loading model.
    model_enum = ModelEnum[config["model_type"]]
    model = get_model_from_enum(model_enum, from_pretrained=bool(config["from_pretrained"]))

    # Running task.
    if config["task"] == "train":
        train_dataset = SuperResolutionData(image_folder=config["paths"]["train_set"],
                                            is_train_split=True,
                                            normalize_lr=True)

        val_dataset = SuperResolutionData(image_folder=config["paths"]["val_set"],
                                          is_train_split=False,
                                          normalize_lr=True)

        # One config file by model type.
        model.train(train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    model_save_folder=os.path.join(config["paths"]['model_save'], config['experiment_name']),
                    images_save_folder=os.path.join(config["paths"]["val_images_save"], config["experiment_name"]),
                    experiment_name=config["experiment_name"],
                    **config["hyperparameters"]
                    )

    if config["task"] == "test":
        if isinstance(config["paths"]["test_set"], str):
            config["paths"]["test_set"] = [config["paths"]["test_set"]]
            config["paths"]["test_images_save"] = [config["paths"]["test_images_save"]]

        for image_folder, images_save_folder in zip(config["paths"]["test_set"], config["paths"]["test_images_save"]):
            test_dataset = SuperResolutionData(image_folder=image_folder,
                                               is_train_split=False,
                                               normalize_lr=True)

            psnr, ssim = model.evaluate(val_dataset=test_dataset,
                                        batch_size=1,
                                        images_save_folder=os.path.join(images_save_folder, config["experiment_name"]),
                                        reverse_normalize=config["reverse_normalize"]
                                        )
            print(f"{os.path.basename(image_folder)} - PSNR: {psnr}, SSIM: {ssim}")

    if config["task"] == 'predict':
        if isinstance(config["paths"]["test_set"], str):
            config["paths"]["test_set"] = [config["paths"]["test_set"]]
            config["paths"]["test_images_save"] = [config["paths"]["test_images_save"]]

        for image_folder, images_save_folder in zip(config["paths"]["test_set"], config["paths"]["test_images_save"]):
            test_dataset = SuperResolutionData(image_folder=image_folder,
                                               is_train_split=False,
                                               normalize_hr=True)

            model.predict(test_dataset=test_dataset,
                          batch_size=1,
                          force_cpu=config["force_cpu"],
                          images_save_folder=os.path.join(images_save_folder, config["experiment_name"])
                          )

    print(f"{config['task']} completed in {time() - start:.2f} seconds.")
