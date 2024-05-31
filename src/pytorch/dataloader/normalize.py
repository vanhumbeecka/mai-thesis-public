from datetime import timedelta
from pkgutil import get_data
from torch.utils.data import DataLoader
from pytorch.config import SolarConfig

from pytorch.dataloader.dl import get_dataloader


def calculate_mean_std(dl: DataLoader):
    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    for data, _, _ in dl:
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_size

    mean /= nb_samples
    std /= nb_samples

    return mean, std


if __name__ == "__main__":
    import os

    local_data_path = "../../../data"
    absolute_path = os.path.dirname(__file__)
    data_path = os.path.join(absolute_path, local_data_path)

    image_data_path = "/mnt/d/datasets/helioviewer/images_resized"
    config: SolarConfig = SolarConfig(
        local_data_path=data_path,
        image_data_path=image_data_path,
        image_dimension=(60, 512, 512),
        channels_name=["source_19"],
        timedimension_max_timedelta=timedelta(hours=12),
        flare_data_file="flare_data.pkl",
    )

    dl, _ = get_dataloader(config, batch_size=8, num_workers=8, shuffle=False)
    mean, std = calculate_mean_std(dl)
    print(mean, std)
