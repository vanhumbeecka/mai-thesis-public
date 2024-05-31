import argparse
import sys
import os
from sympy import timed
import torch

import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import List, Tuple

from common.utils import init_logger
from DONKI.pytorch_dataset import HelioViewerDataset
from pytorch.config import FlareEvent, SolarConfig
import torchvision.transforms as transforms
import pytorch.dataloader.dl as dl
from dataclasses import dataclass
from functools import lru_cache


logger = init_logger(__name__)


def transform_solar_tensor(t: torch.Tensor):
    image_t = t.float() / 255.0  # float32
    return image_t


def get_train_image_set(
    dataset: HelioViewerDataset, flare_idx: int
) -> Tuple[torch.Tensor, str, List[str]]:
    channel_idx = 0  # only 1
    tensors, label, filenames = dataset[flare_idx]  # no batchsize with dataset
    t: torch.Tensor = tensors[channel_idx]

    # return torch.reshape(t, (3072, 512)), label
    return t, label, filenames[0] # only first channel filenames (hardcoded)


# @lru_cache(maxsize=1)
def get_dataloader(
    config: SolarConfig, batch_size: int, num_workers: int, shuffle: bool = True, data_only: bool = False, pin_memory: bool = False
) -> Tuple[DataLoader, HelioViewerDataset]:
    """Initialize a DataLoader with the given parameters."""
    logger.info(f"Loading annotations from {config.data_path}")
    flare_list: List[FlareEvent] = config.get_annotations()

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img_np: torch.from_numpy(img_np)),  # uint8
            transforms.Resize((512, 512), antialias=True),  # type: ignore
            transforms.Lambda(
                lambda x: x.float() / 255.0
            ),  # normalize between 0 and 1 (BCE needs values between 0 and 1)
        ]
    )

    dataset = HelioViewerDataset(
        flare_data=flare_list,
        img_dir=config.get_image_dir(),
        sources=config.channels_name,
        depth_dimension=config.depth_dimension,
        max_timedelta=config.timedimension_max_timedelta,
        transform=transform,
        data_only=data_only
    )

    return (
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        ),
        dataset,
    )


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../../../data/")
    # image_path = '/mnt/d/datasets/helioviewer/images_resized/'
    image_path = os.path.join(data_path, "flare_images_preprocessed")
    print(image_path)
    batch_size = 4
    num_channels = 1

    config: SolarConfig = SolarConfig(
        local_data_path=data_path,
        image_data_path=image_path,
        image_dimension=(6, 512, 512),
        channels_name=["source_19"],
        timedimension_max_timedelta=timedelta(hours=16),
        flare_data_file="valid_flares_10h.pickle",
    )
    dataloader, _ = dl.get_dataloader(config, batch_size, 8, shuffle=True)

    dataloader_iter = iter(dataloader)
    images, labels, flare_names_by_source = next(dataloader_iter)

    print("min:", images.min())
    print("max:", images.max())

    print("shape:\t", images.shape)
    print("labels:\t", labels)
    print("flare_names_by_source:\t", flare_names_by_source)

    assert images.shape == torch.Size([batch_size, num_channels, 6, 512, 512])
