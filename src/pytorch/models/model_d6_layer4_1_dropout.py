import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple
import numpy as np
import os

# https://poloclub.github.io/cnn-explainer/


class ModelDepth6Layer4Dropout(nn.Module):
    def __init__(self, in_chan=1, latent_size=10, dropout=0.0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_chan, 6, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=0
            ),  # Assuming single-channel input
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.Conv3d(6, 12, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.Conv3d(12, 24, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=0),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.reshape = (24, 1, 31, 31)

        self.fc_mu = nn.Linear(int(np.prod(self.reshape)), latent_size)
        self.fc_logvar = nn.Linear(int(np.prod(self.reshape)), latent_size)

        self.decoder_fc = nn.Linear(latent_size, int(np.prod(self.reshape)))

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                24, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0
            ),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.ConvTranspose3d(
                24, 12, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=0
            ),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.ConvTranspose3d(12, 6, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0),
            nn.Dropout3d(p=dropout),
            nn.ReLU(),
            nn.ConvTranspose3d(
                6, in_chan, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=0
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def decode(self, z):
        recon = self.decoder_fc(z)
        recon = recon.view(-1, *self.reshape)
        return self.decoder(recon)

    def forward(self, input_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(input_batch)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)

        return x_reconstructed, mu, logvar


def load_from_version_70(pl_logs_path: str, dropout=0.0) -> ModelDepth6Layer4Dropout:
    """Load a model from version 70.
    This model is of type ModelDepth6Layer4 and should be converted to ModelDepth6Layer4Dropout.
    """
    m = ModelDepth6Layer4Dropout(latent_size=32, dropout=dropout)

    path = os.path.join(
        pl_logs_path, "version_70", "checkpoints", "epoch=49-step=2450.ckpt"
    )
    checkpoint_data = torch.load(path)
    state_dict = checkpoint_data["state_dict"]

    # remove prefix 'model.' from keys
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    new_state_dict = {
        k.replace("encoder.6.", "encoder.9.")
        .replace("encoder.2.", "encoder.3.")
        .replace("encoder.4.", "encoder.6."): v
        for k, v in new_state_dict.items()
    }
    new_state_dict = {
        k.replace("decoder.6.", "decoder.9.")
        .replace("decoder.2.", "decoder.3.")
        .replace("decoder.4.", "decoder.6."): v
        for k, v in new_state_dict.items()
    }
    m.load_state_dict(new_state_dict)

    return m


def load_from_version(
    pl_logs_path: str,
    version: int = 105,
    checkpoint_file: str = "epoch=45-step=17756.ckpt",
    dropout=0.0,
) -> ModelDepth6Layer4Dropout:
    """Load a model from a specific version and checkpoint file.
    Assume the model is of type ModelDepth6Layer4Dropout"""
    m = ModelDepth6Layer4Dropout(latent_size=32, dropout=dropout)
    path = os.path.join(
        pl_logs_path, f"version_{version}", "checkpoints", checkpoint_file
    )
    checkpoint_data = torch.load(path)
    state_dict = checkpoint_data["state_dict"]

    # remove prefix 'model.' from keys
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    m.load_state_dict(new_state_dict)

    return m


def test_load_from_version_70() -> None:
    # pl_logs_path = '../../../data/lightning_logs'
    cur = os.getcwd()
    print(cur)
    pl_logs_path = os.path.join(cur, "data/lightning_logs")
    model = load_from_version_70(pl_logs_path, dropout=0.0)


if __name__ == "__main__":
    # Create a random input tensor
    input_channel_size = 1
    latent_size = 32
    input_shape = (6, 512, 512)
    input_tensor = torch.randn(
        1, input_channel_size, *input_shape
    )  # Batch size of 1, 3 channels, 32x32 image
    model = ModelDepth6Layer4Dropout(input_channel_size, latent_size=latent_size)
    batch_size = 16

    summary(model, (batch_size, input_channel_size, *input_shape), device="cpu")

    recon, mu, logvar = model(input_tensor)
    assert input_tensor.shape == recon.shape
    print(model)
