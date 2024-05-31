import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from typing import Tuple

class SimpleModel2(nn.Module):
    def __init__(self, latent_size: int, input_shape: Tuple[int, int, int] = (60, 512, 512)):
        super(SimpleModel2, self).__init__()

        self.depth, self.height, self.width = input_shape
        self.padding = 2

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=5, padding=self.padding),  # Assuming single-channel input
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=5, stride=5, padding=self.padding),
            nn.ReLU(),
            nn.Flatten()
        )

        self.reshape = (32, (self.depth // 8), (self.height // 8), (self.width // 8))

        self.fc_mu = nn.Linear(int(np.prod(self.reshape)), latent_size)
        self.fc_logvar = nn.Linear(int(np.prod(self.reshape)), latent_size)

        self.decoder_fc = nn.Linear(latent_size, int(np.prod(self.reshape)))

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=5, stride=5, padding=self.padding),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=5, stride=5, padding=self.padding),
            nn.Sigmoid()
        )

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
        x_reconstructed = self.decoder(x_reconstructed)

        return x_reconstructed, mu, logvar


if __name__ == '__main__':
    from torchinfo import summary
    from torchvision.transforms import ToTensor

    # Example usage:
    input_shape = (60, 512, 512)
    batch_size = 8
    input_channel_size = 1
    latent_size = 16

    model = SimpleModel2(input_shape=input_shape, latent_size=latent_size)
    summary(model, (batch_size, input_channel_size, *input_shape), device='cpu')

    # Assuming you have a 3D tensor as input_data
    size = (1, 1, 60, 512, 512) # (batch_size, channels, depth, height, width)
    input_data = torch.randint(0, 255, size, dtype=torch.uint8)
    output, mu, logvar = model(input_data / 255.0)
    print(input_data.shape, output.shape)
    # print(output, mu, logvar)
