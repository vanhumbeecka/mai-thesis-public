from matplotlib.pylab import f
import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
from typing import Tuple, List, Dict, Union, Any, cast, Literal

from zmq import device
from pytorch.dataloader.dl import get_dataloader, FlareEvent
from pytorch.config import SolarConfig
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import logging


# from mnist.model import VAEMNIST
from DONKI.pytorch_dataset import HelioViewerDataset
from pytorch.dataloader.dl import get_dataloader
from pytorch.training_loop import parser
from pytorch.models.model_d6_layer4_1 import ModelDepth6Layer4
from pytorch.models.model_d6_layer4_1_dropout import (
    ModelDepth6Layer4Dropout,
    load_from_version,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_config(data_path: str) -> SolarConfig:

    image_path = os.path.join(data_path, "flare_images_preprocessed")
    pl_logs = os.path.join(data_path, "lightning_logs")
    print("data_path: ", data_path)
    print("pl_logs: ", pl_logs)

    config: SolarConfig = SolarConfig(
        local_data_path=data_path,
        image_data_path=image_path,
        image_dimension=(6, 512, 512),
        channels_name=["source_19"],
        timedimension_max_timedelta=timedelta(hours=16),
        flare_data_file="valid_flares_10h.pickle",
    )

    return config


def init_data(
    config: SolarConfig, batch_size: int
) -> Tuple[DataLoader, HelioViewerDataset]:
    num_workers = 8
    dataloader, dataset = get_dataloader(
        config, batch_size, num_workers=num_workers, shuffle=False
    )
    return dataloader, dataset


def load_model_version(logs_dir: str, version: int, checkpoint_file: str):
    version = 112
    checkpoint_file = "epoch=11-step=4632.ckpt"

    model: ModelDepth6Layer4Dropout = load_from_version(
        logs_dir, dropout=0.0, version=version, checkpoint_file=checkpoint_file
    )
    return model


def load_kl_losses(logs_dir: str, version: int, latent_dim: int = 32) -> Dict[str, float]:
    latent_kl_loss_path = os.path.join(
        logs_dir, f"version_{version}", "latent_kl_loss.txt"
    )

    # read last 32 lines of the file (32 latent dimensions)
    with open(latent_kl_loss_path) as file:
        last_lines = file.readlines()[-latent_dim:]
    kl_losses = {l.split(",")[2]: float(l.split(",")[3].strip()) for l in last_lines}
    return kl_losses


def get_average_latent_activation(
    m,
    data_loader: DataLoader,
    latent_dim: int = 32,
    device: Literal["cpu", "cuda"] = "cuda",
):
    """Returns tensor of shape [latent_dim, 1, 6, 512, 512]"""
    average_z = torch.zeros((latent_dim, 1, 6, 512, 512), device=device)
    count = 0

    for data, labels, filenames in tqdm(data_loader):
        # data shape (16, 1, 6, 512, 512) = (batch_size, channels, time, height, width)
        # print(data.shape)
        data = data.to(device)
        batch_size = data.size(0)

        with torch.inference_mode():
            recon, mu, log_var = m(data)

        std = torch.exp(0.5 * log_var)
        z = mu + std  # shape [16, 32] = (batch_size, latent_dim)
        assert latent_dim == z.size(
            1
        ), f"Expected latent_dim {latent_dim} but got {z.size(1)}"

        for b in range(batch_size):
            z_batch_item = z[b]  # shape [32]
            img = data[b]  # shape [1, 6, 512, 512]
            for z_index in range(latent_dim):
                z_val = z_batch_item[z_index]
                average_z[z_index] += z_val * img

        count += batch_size

    average_z /= count

    return average_z


def load_average_latent_activation(
    m,
    model_version: int,
    data_loader: DataLoader,
    latent_dim=32,
    device: Literal["cuda", "cpu"] = "cuda",
) -> torch.Tensor:
    # Check if file exists
    file_path = f"average_tensor_v{model_version}.pt"
    if not os.path.exists(file_path):
        # Save average_tensor to file
        average_tensor = get_average_latent_activation(
            m, data_loader, latent_dim=latent_dim, device=device
        )
        torch.save(average_tensor, file_path)
        print("Saved average_tensor to", file_path)
    else:
        # Load average_tensor from file
        average_tensor = torch.load(file_path)
        print("Loaded average_tensor from", file_path)

    return average_tensor


def get_model_latent_mean_traversal_grid(
    model, latent_dim: int, kl_losses: Dict[str, float], z_sample: torch.Tensor
):
    assert z_sample.shape == (
        1,
        latent_dim,
    ), f"Expected shape (1, {latent_dim}) but got {z_sample.shape}"

    # calculate adapted z = mu + std * eps where eps ~ N(0,1)
    # iterate over -3, +3 std
    kl_losses_num = [kl_losses[k] for k in kl_losses]
    sorted_indexes = np.argsort(kl_losses_num)[::-1].tolist()
    sorted_var = [kl_losses_num[s] for s in sorted_indexes]

    # init z
    grid = torch.zeros(
        (latent_dim, 7, 1, 6, 512, 512)
    )  # <latent_dim, x-axis, channels, time, height, width>
    std_space = np.linspace(-3, 3, 7)
    for i, z_index in tqdm(enumerate(sorted_indexes), total=latent_dim):
        for idx, s in enumerate(std_space):
            # z_clone = torch.zeros((1, latent_dim))
            z_clone = z_sample.clone()
            z_clone[0][z_index] = s  # shape [1, latent_dim]
            recon_z = model.decode(z_clone).clone()
            grid[i][idx] = recon_z

    # save_image(grid.view(10 * 7, 1, 32, 32), 'mnist/results/beta1_00.png', nrow=7)

    return grid, sorted_indexes, sorted_var


def plot_all(
    grid,
    sorted_indexes,
    sorted_var,
    average_tensor: torch.Tensor,
    beta: float,
    plot_time_dimension: int = 5,
    suffix: str = "",
):
    average_tensor = (
        average_tensor.to("cpu") if average_tensor.is_cuda else average_tensor
    )
    # grid = grid.view(32, 7, 1, 6, 512, 512)
    h = grid.shape[0]
    w = grid.shape[1]
    img_height = grid.shape[-1]
    latent_dim = grid.shape[0]

    # Define the labels for the x-ticks

    x_ticks_labels = [
        x for x in [f"-3", "-2", "-1", r"$\leftarrow$0$\rightarrow$", "1", "2", "3"]
    ]  # Replace this with your actual labels
    x_ticks_labels = [f"{l}\n$t_{{{plot_time_dimension}}}$" for l in x_ticks_labels]
    y_ticks_labels_1 = [f"$z_{{{x}}}$" for x in sorted_indexes]
    y_ticks_labels_2 = [f"{x:.2f}" for x in sorted_var]
    y_ticks_labels_zip = [
        j + "\nKLD=" + i for i, j in zip(y_ticks_labels_2, y_ticks_labels_1)
    ]

    extra_width = 6
    extra_width_avg = 1
    fig, ax = plt.subplots(
        h,
        w + extra_width + extra_width_avg,
        figsize=(w + extra_width, h),
        constrained_layout=True,
    )
    for i in tqdm(range(h)):  # height
        for j in range(
            w + extra_width
        ):  # columns: first 7 are the std spread, next 6 are the average, last one is the average of the average

            if j < w:
                # grid[0][0].permute(1, 2, 3, 0)[<time_dim>] shape equals <512, 512, 1>
                ax[i, j].imshow(
                    grid[i][j].permute(1, 2, 3, 0)[plot_time_dimension].detach().numpy(),
                    cmap="grey",
                )

                if i == h - 1:
                    ax[i, j].set_xticks([img_height / 2])
                    ax[i, j].set_xticklabels([x_ticks_labels[j]], fontsize="medium")
                    ax[i, j].tick_params(axis="x", which="both", length=0)
                else:
                    ax[i, j].get_xaxis().set_visible(False)
                if j == 0:
                    ax[i, j].set_yticks([img_height / 2])
                    ax[i, j].set_yticklabels([y_ticks_labels_zip[i]], fontsize="medium")
                    ax[i, j].tick_params(axis="y", which="both", length=0)
                else:
                    ax[i, j].get_yaxis().set_visible(False)

            else:
                idx = sorted_indexes[i]
                time_idx = j - w
                ax[i, j].imshow(
                    average_tensor[idx].squeeze().detach().numpy()[time_idx], cmap="jet"
                )
                ax[i, j].get_yaxis().set_visible(False)

                if i == h - 1:
                    ax[i, j].set_xticks([img_height / 2])
                    ax[i, j].set_xticklabels([f"$t_{{{time_idx}}}$"], fontsize="medium")
                    ax[i, j].tick_params(axis="x", which="both", length=0)
                else:
                    ax[i, j].get_xaxis().set_visible(False)

        # plot average of average_tensor for each i
        idx = sorted_indexes[i]
        ax[i, w + extra_width].imshow(
            average_tensor[idx].squeeze().detach().numpy().mean(axis=0), cmap="jet"
        )
        ax[i, w + extra_width].get_yaxis().set_visible(False)
        ax[i, w + extra_width].spines["bottom"].set_linewidth(1.5)
        ax[i, w + extra_width].spines["top"].set_linewidth(1.5)
        ax[i, w + extra_width].spines["right"].set_linewidth(1.5)
        ax[i, w + extra_width].spines["left"].set_linewidth(1.5)

        if i == h - 1:
            ax[i, w + extra_width].set_xticks([img_height / 2])
            ax[i, w + extra_width].set_xticklabels(
                ["Average\nactivation\nin time"], fontsize="medium"
            )
            ax[i, w + extra_width].tick_params(axis="x", which="both", length=0)
        else:
            ax[i, w + extra_width].get_xaxis().set_visible(False)

    fig.suptitle(
        f"Latent space traversal ($\\beta$ = {beta}, latent space dimension = {latent_dim})",
        ha="center",
        va="bottom",
    )
    plt.savefig(f"latent_space_traversal_beta_{beta}{suffix}.png", bbox_inches="tight")


if __name__ == "__main__":
    import argparse
    import os
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../../data")
    # parser.add_argument("--version", type=int, default=112)
    parser.add_argument("--version", type=int, default=128)
    # parser.add_argument("--checkpoint-file", type=str, default="epoch=11-step=4632.ckpt")
    parser.add_argument("--checkpoint-file", type=str, default="epoch=99-step=38600.ckpt")
    parser.add_argument("--beta", type=float, default=20.0, help="Beta for KL loss")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    absolute_path = os.path.dirname(__file__)
    data_path = os.path.join(absolute_path, args.data_path)

    # params
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl_logs = os.path.join(data_path, "lightning_logs")

    # init
    config = init_config(data_path)
    dataloader, dataset = init_data(config, batch_size=args.batch_size)
    kl_losses = load_kl_losses(pl_logs, version=args.version)
    model = load_model_version(
        pl_logs, version=args.version, checkpoint_file=args.checkpoint_file
    )
    model.to(device)

    # calculate
    average_tensor = load_average_latent_activation(
        model,
        args.version,
        dataloader,
        latent_dim=args.latent_dim,
        device=device,
    )

    # Get a sample
    sample, label, filenames = dataset[1]
    sample = sample.to(device)
    with torch.inference_mode():
        batch_sample = sample.unsqueeze(0)
        mu, logvar = model.encode(batch_sample)

    # apply kl_losses to sample in order to plot the grid
    grid, sorted_indexes, sorted_var = get_model_latent_mean_traversal_grid(
        model, latent_dim=32, kl_losses=kl_losses, z_sample=mu
    )

    # plot
    plot_all(
        grid,
        sorted_indexes,
        sorted_var,
        average_tensor,
        beta=args.beta,
        suffix=f"_v{args.version}",
    )
