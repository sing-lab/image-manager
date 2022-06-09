"""Utility functions for super resolution"""
import torch
import numpy as np
import random
import os
from typing import Tuple, Optional
from super_resolution_data import SuperResolutionData
from torch.utils.data import DataLoader
from torch import tensor
import yaml


def set_seed(seed: int = 0):
    """
    Set seed for reproducibility

    Parameters
    ----------
    seed: int
        The seed value.

    Returns
    -------

    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    """

    Parameters
    ----------
    worker_id

    Returns
    -------

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_stat(dataset: Optional[SuperResolutionData] = None, dataset_name: Optional[str] = None) \
        -> Tuple[tensor, tensor]:
    """
    Compute an RGB image dataset mean and standard deviation for normalization purpose.

    Parameters
    ----------
    dataset: SuperResolutionData, default None
        The dataset to be analyzed.
    dataset_name: str, default None
        The name of a dataset, in ("ImageNet")

    Returns
    -------
    Tuple[tensor, tensor]
        List of mean for each channel, list of standard deviation for each channel.

    Raises
    ------
    ValueError
        If 'dataset_name' not in 'ImageNet'.

    """
    if dataset_name is not None and dataset_name.lower() != "imagenet":
        raise ValueError(f"Parameter 'dataset_name' should be 'ImageNet' but is {dataset_name}")

    if dataset_name is not None and dataset_name.lower() == "imagenet":
        return torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    total_batch = len(data_loader)

    channels_sum, channels_squared_sum, num_batches = 0., 0., 0.
    for i_batch, (_, hr_images) in enumerate(data_loader):
        print(f'{i_batch + 1}/{total_batch}\r', end="")

        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(hr_images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(hr_images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def load_config(config_path):
    """
    Load a configuration file

    Parameters
    ----------
    config_path: str
        The config file path.

    Returns
    -------

    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config
