import argparse
import sys
import os

import pandas as pd
import numpy as np
import argparse
import json
from requests import get
from datetime import datetime, timedelta

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from common.utils import init_logger, enumerate_with_estimate
from DONKI.pytorch_dataset import HelioViewerDataset
from pytorch.config import SolarConfig
import pytorch.dataloader.dl as dl
from pytorch.dataloader.dl import FlareEvent


logger = init_logger(__name__)


def main(config: SolarConfig, batch_size: int, num_workers: int):
    loader, _ = dl.get_dataloader(config, batch_size, num_workers)

    batch_iter = enumerate_with_estimate(loader, "populating cache", num_workers)

    for _ in batch_iter:
        pass


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../../data/")
    image_path = os.path.join(data_path, "flare_images_preprocessed")
    print(os.path.abspath(image_path))
    batch_size = 4

    config: SolarConfig = SolarConfig(
        local_data_path=data_path,
        image_data_path=image_path,
        image_dimension=(6, 512, 512),
        channels_name=["source_19"],
        timedimension_max_timedelta=timedelta(hours=16),
        flare_data_file="valid_flares_10h.pickle",
    )

    main(
        config=config,
        batch_size=8,
        num_workers=8,
    )
