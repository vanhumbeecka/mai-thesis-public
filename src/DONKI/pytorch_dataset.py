import functools
from venv import logger
from torch.utils.data import Dataset
import os
import pandas as pd
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.cuda
from common.disk import getCache

from common.utils import init_logger
from common.data_clean import filter_by_clean_data
from datetime import timedelta
import torchvision.transforms as transforms

from pytorch.config import FlareEvent

logger = init_logger(__name__)

raw_cache = getCache("source-raw")
# memory = getMemCache("source-raw")


@raw_cache.memoize(typed=True)
def get_flare_array_single_source(
    flare_id_path: str, source: str
) -> Tuple[np.ndarray, List[str]]:
    """This will use cached data if available. If not, it will read the raw data and cache it."""
    channel_set: List[np.ndarray] = []
    folder_path = os.path.join(flare_id_path, source)
    jp2_files = sorted(
        [
            file
            for file in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, file))
        ]
    )

    for file in jp2_files:
        image_path = os.path.join(folder_path, file)
        img = Image.open(image_path)
        channel_set.append(np.array(img))
    return np.stack(channel_set, axis=0), jp2_files


def space_evenly_indices(array_size, num_indices):
    # Calculate the step size to evenly distribute indices
    step_size = array_size / (num_indices + 1)

    # Generate indices by rounding equidistant positions
    indices = [int((i + 1) * step_size) for i in range(num_indices)]

    return indices


def get_flare_array(
    flare_id_path: str, sources: List[str], depth_dimension: int
) -> Tuple[np.ndarray, List[List[str]]]:
    data_set: List[np.ndarray] = []
    file_set: List[List[str]] = []
    for source in sources:
        try:
            arr, files = get_flare_array_single_source(
                flare_id_path, source
            )  # full range (60 depth)
            depth = arr.shape[0]
            indices = space_evenly_indices(depth, depth_dimension)
            arr = arr[indices]
            files = [files[i] for i in indices]
            data_set.append(arr)
            file_set.append(files)
        except Exception as e:
            logger.error(f"Error getting image for {flare_id_path}: {str(e)}")
            raise e
    return np.stack(data_set, axis=0), file_set


class HelioViewerDataset(Dataset):
    def __init__(
        self,
        flare_data: List[FlareEvent],
        img_dir: str,
        sources: List[str] = ["source_19"],
        depth_dimension: int = 60,
        max_timedelta: timedelta = timedelta(seconds=3500),
        transform=None,
        data_only: bool = False,
    ) -> None:
        super().__init__()
        self.flare_data = flare_data
        self.img_dir = img_dir
        self.sources = sources
        self.depth_dimension = depth_dimension
        self.max_timedelta = max_timedelta
        self.data_only = data_only

        if transform is None:
            logger.info("Using default transform")
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(lambda img_np: torch.from_numpy(img_np)),
                    transforms.Lambda(lambda x: x.float() / 255.0),
                ]
            )
        else:
            self.transform = transform

    @functools.cached_property
    def _get_flare_ids(self) -> List[str]:
        # filenames = [flare_id.replace(':', '_') for flare_id in self.flare_data.index.tolist()]
        filtered_results = filter_by_clean_data(
            self.img_dir,
            self.flare_data,
            expected_time_dimension=self.depth_dimension,
            expected_max_timedelta=self.max_timedelta,
        )
        # filtered_results = [flare_id.replace("_", ":") for flare_id in filtered_results]
        return filtered_results

    def __len__(self) -> int:
        return len(self._get_flare_ids)

    def __getitem__(self, idx: int):
        """Returns an image (ndarray) or a Tuple[image, label, filename] if data_only is False.

        Returns:
            Tuple[np.ndarray, str, List[List[str]]] | np.ndarray
        """
        flare_id = self._get_flare_ids[idx]
        img_path = os.path.join(self.img_dir, flare_id)
        flare_np, flare_names_by_source = get_flare_array(
            img_path, self.sources, self.depth_dimension
        )

        label = [f for f in self.flare_data if f.archive_id.split("/")[-1] == flare_id][
            0
        ].flare_class
        image_t = self.transform(flare_np)

        if self.data_only:
            return image_t
        return image_t, label, flare_names_by_source


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(".")

    data_path = os.path.join(project_dir, "data")
    img_dir = "/mnt/d/datasets/helioviewer/images_resized"

    annotations_file = os.path.join(data_path, "flare_data.pkl")
    df = pd.read_pickle(annotations_file)

    logger.info(f"Project directory: {os.path.abspath('.')}")
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Image directory: {img_dir}")

    filtered_df = df[(df.index >= "2012-01-01") & (df.index <= "2013-01-01")]
    dataset = HelioViewerDataset(
        flare_data=filtered_df, img_dir=img_dir, data_only=False
    )
    logger.info(f"Dataset length: {len(dataset)}")
    img, label, filenames = dataset[0]
    logger.info(f"Image shape: {img.shape}")
    img, label, filenames = dataset[0]  # check if 2nd time is faster retrieval
    logger.info(f"Image shape: {img.shape}")

    logger.info((img.dtype, img.min(), img.max()))

    logger.info(label)
