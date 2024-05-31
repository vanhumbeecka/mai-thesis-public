from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from typing import Optional


class LitAutoEncoder(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        beta: float,
        learning_rate: float,
        weight_decay: float,
        gamma: float,
    ):
        super().__init__()
        self.model = model
        self.beta = beta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma

        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.learning_rate > 0, "Learning rate must be positive"

        # Define your optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Define the learning rate scheduler
        self.lr_scheduler = None
        if self.gamma is not None:
            print(f"Using StepLR learning rate scheduler with gamma={self.gamma}")
            self.lr_scheduler = StepLR(
                self.optimizer, step_size=1, gamma=self.gamma, verbose=False
            )
        else:
            print("No learning rate scheduler is defined")

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y, filenames = batch
        reconstructed, mu, log_var = self.model(x)
        total_loss, BCE, KLD = self.loss(reconstructed, x, self.beta, mu, log_var)
        self.log_dict(
            {
                "total_loss": total_loss,
                "BCE_loss": BCE,
                "KLD_loss": KLD,
                "BCE_avg_loss": BCE / x.size(0),
                "KLD_avg_loss": KLD / x.size(0),
                "BCE_TOTAL_ratio": BCE / total_loss,
                "learning_rate": self.learning_rate
            }
        )
        return total_loss

    def loss(self, recon_x, x, beta, mu, lvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * beta * torch.sum(1 + lvar - mu.pow(2) - lvar.exp())
        return BCE + KLD, BCE, KLD

    def configure_optimizers(self):
        # Return the optimizer and scheduler (if any)
        if self.lr_scheduler is None:
            return self.optimizer

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",  # Adjust learning rate after every epoch
                "frequency": 1,
            },
        }
