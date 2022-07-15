"""utility functions for models"""
from torch import tensor
from math import ceil
import torch
from torchvision.transforms import Pad


def get_tiles_from_image(image: tensor, tile_size: int, tile_overlap: int):
    """
    Split an input image into tiles. Assume original image is squared.
    Pad original image to get same size tiles.

    Parameters
    ----------
    image: tensor
        Input image to be split into tiles.
    tile_size: int
        Size of each tile (a tile is squared).
    tile_overlap: int
        Overlap pixels between tiles.

    Returns
    -------
    tensor
        Input image split into tiles

    """
    channel, height, width = image.shape

    tiles_x = ceil(width / tile_size)
    tiles_y = ceil(height / tile_size)

    # 1st padding: image width and height should be multiple of tile_size #TODO prendre le max des pad des deux ( + 1 si pas divisible par deux ?)?
    pad_x = ceil((tiles_x * tile_size - width) / 2)
    pad_y = ceil((tiles_y * tile_size - height) / 2)

    # Reflects helps the model to improve super res
    image = Pad((pad_x, pad_y, pad_x, pad_y), padding_mode="reflect")(image)

    # 2nd padding: add overlap size to each image side
    image = Pad((tile_overlap, tile_overlap, tile_overlap, tile_overlap), padding_mode="reflect")(image)

    _, height, width = image.shape

    # Output : tiles_number * batch_size * channel * tile_size * tile_size
    output = torch.empty((tiles_x * tiles_y, channel, tile_size + 2 * tile_overlap, tile_size + 2 * tile_overlap))

    for index_y in range(tiles_y):
        for index_x in range(tiles_x):
            tile_index = index_y * tiles_x + index_x

            # Extract tile from input image
            offset_x = index_x * tile_size + tile_overlap  # First tile from the left: start at tile_overlap
            offset_y = index_y * tile_size + tile_overlap

            # Input tile area on total image
            input_start_x = offset_x
            input_end_x = min(offset_x + tile_size, width)
            input_start_y = offset_y
            input_end_y = min(offset_y + tile_size, height)

            # Input tile area on total image with overlapping.
            input_start_x_pad = max(input_start_x - tile_overlap, 0)
            input_end_x_pad = min(input_end_x + tile_overlap, width)
            input_start_y_pad = max(input_start_y - tile_overlap, 0)
            input_end_y_pad = min(input_end_y + tile_overlap, height)

            # Input tile dimensions
            output[tile_index] = image[:, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

    return output


def get_image_from_tiles(sr_tiles: tensor, tile_size: int, tile_overlap: int, scaling_factor: int,
                         image: tensor) -> tensor:
    """
    Merge upscaled tiles to get an upscaled image.

    Parameters
    ----------
    sr_tiles: tensor
        Super resolved tiles.
    tile_size: int
        Size of each tile (a tile is squared), before applying super resolution.
    tile_overlap: int
        Overlap pixels between tiles, before applying super resolution.
    scaling_factor: int
        The scaling factor to use when downscaling high resolution images into low resolution images.
    image: tensor
        Original image to be upscaled.

    Returns
    -------
    tensor
        The super resolution image.
    """
    channel, height, width = image.shape
    tiles_x = ceil(width / tile_size)
    tiles_y = ceil(height / tile_size)

    # Start with a black image
    sr_images = image.new_zeros((channel, tiles_y * tile_size * scaling_factor,
                                 tiles_x * tile_size * scaling_factor))

    for index_tile, tile in enumerate(sr_tiles):
        # Retrieve cropped tile from padded tile.
        tile_cropped = tile[:,
                       tile_overlap * scaling_factor:(tile_overlap + tile_size) * scaling_factor,
                       tile_overlap * scaling_factor:(tile_overlap + tile_size) * scaling_factor]

        index_tile_x = index_tile % tiles_x
        index_tile_y = index_tile // tiles_x

        sr_images[:,
        index_tile_y * tile_size * scaling_factor: (index_tile_y + 1) * tile_size * scaling_factor,
        index_tile_x * tile_size * scaling_factor: (index_tile_x + 1) * tile_size * scaling_factor
        ] = tile_cropped

    # Remove padding
    pad_x = scaling_factor * ceil((tiles_x * tile_size - width) / 2)
    pad_y = scaling_factor * ceil((tiles_y * tile_size - height) / 2)

    if pad_x:
        sr_images = sr_images[:, :, pad_x:-pad_x]  # Remove padding
    if pad_y:
        sr_images = sr_images[:, pad_y:-pad_y, :]  # Remove padding

    sr_images = torch.unsqueeze(sr_images, 0)  # Add the batch dimension

    return sr_images
