from venv import logger
import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple
import numpy as np
from common.utils import init_logger
from pytorch.models.base import BaseModel

logger = init_logger(__name__)

# https://poloclub.github.io/cnn-explainer/

class ModelDepth6Layer4V2(BaseModel):
    def __init__(self, in_chan: int, latent_size: int, dropout: float, init_weights: bool):
        super().__init__()

        self.reshape = (24, 1, 31, 31)

        # Encoder
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(int(np.prod(self.reshape)), 4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Conv3d(in_chan, 6, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=0),  # Assuming single-channel input
            nn.ReLU(),
            nn.Conv3d(6, 12, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.Conv3d(12, 24, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.Flatten(),
            self.fc1,
            self.fc2
        )
        
        self.fc_mu = nn.Linear(4096, latent_size)
        self.fc_logvar = nn.Linear(4096, latent_size)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size, 4096),
            nn.ReLU()
        )
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.decoder_fc2 = nn.Sequential(
            nn.Linear(4096, int(np.prod(self.reshape))),
            nn.ReLU(),
        )

        self.decoder_fc = nn.Sequential(
            self.decoder_fc,
            self.decoder_fc1,
            self.decoder_fc2,
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose3d(24, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(24, 12, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(12, 6, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(6, in_chan, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=0),
            nn.Sigmoid()
        )

        # init weights
        if init_weights:
            logger.info("Initializing weights")
            self._init_weights()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_batch.shape[0]
        x = self.encoder(input_batch)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        z = self.reparameterize(mu, logvar)

        x_reconstructed = self.decoder_fc(z)
        x_reconstructed = x_reconstructed.view(batch_size, *self.reshape)
        x_reconstructed = self.decoder_cnn(x_reconstructed)
        # x_reconstructed = self.decoder_fc(z)
        

        return x_reconstructed, mu, logvar


if __name__ == "__main__":
    # Create a random input tensor
    input_channel_size = 1
    latent_size = 16
    input_shape = (6, 512, 512)
    input_tensor = torch.randn(1, input_channel_size, *input_shape)  # Batch size of 1, 3 channels, 32x32 image
    model = ModelDepth6Layer4V2(input_channel_size, latent_size=latent_size, dropout=0.5, init_weights=True)
    batch_size = 4

    summary(model, (batch_size, input_channel_size, *input_shape), device='cpu')

    recon, mu, logvar = model(input_tensor)
    print(input_tensor.shape, recon.shape)
    assert input_tensor.shape == recon.shape
