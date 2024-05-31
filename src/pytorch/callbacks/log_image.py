from lightning import Callback, LightningModule, Trainer
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from typing import List


class ImageStoreCallback(Callback):
    def __init__(self, input_set, every_n_epochs=1):
        super().__init__()
        self.input_set = input_set.unsqueeze(0)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_set.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs, mu, logvar = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(
                0, 1
            )  # shape (2, 1, 6, 512, 512)
            combined = self.create_image(imgs)

            for c in range(len(combined)):
                channel_data = combined[c]
                trainer.logger.experiment.add_image(
                    f"Reconstructions channel {c+1}",
                    channel_data,
                    global_step=trainer.global_step,
                    dataformats="HW",
                )

    def create_image(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        combined_images = []
        for channel in range(tensor.shape[1]):
            channel_images = []
            for i in range(tensor.shape[0]):
                row_images = []
                for j in range(tensor.shape[2]):
                    row_images.append(tensor[i, channel, j])
                # Concatenate row images horizontally
                channel_images.append(torch.cat(row_images, dim=1))
            # Concatenate row images vertically
            combined_images.append(torch.cat(channel_images, dim=0))

        return combined_images
