from turtle import forward
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

from common.utils import init_logger

logger = init_logger(__name__)


class FullDiskModel(nn.Module):
    def __init__(self, img_size=(1, 60, 512, 512), latent_dim=10):
        """
        Class which defines model and forward pass.
        Args:
            img_size: tuple, (channels, height, width)
            latent_dim: int, size of latent dimension
        """
        super(FullDiskModel, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2] * self.img_size[3]
        self.encoder = FullDiskEncoder(in_channels=1, latent_dim=self.latent_dim)
        self.decoder = FullDiskDecoder(out_channels=1, latent_dim=self.latent_dim)

        # self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def forward(self, input_batch):
        # Encode
        mu, log_var = self.encoder(input_batch)

        # Reparameterize
        z = self.reparametrize(mu, log_var)

        # Decode
        out = self.decoder(z)
        return out, mu, log_var

    
class FullDiskEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, bias=False) -> None:
        super().__init__()

        self.block1 = FullDiskBlock2C(in_channels, conv_channels=64, bias=bias)
        self.block2 = FullDiskBlock2C(64, conv_channels=128, bias=bias)
        self.block3 = FullDiskBlock3C(128, conv_channels=256, bias=bias)
        self.block4 = FullDiskBlock3C(256, conv_channels=512, bias=bias)
        self.block5 = FullDiskBlock3C(512, conv_channels=512, bias=bias)

        # flatten
        self.fc = nn.Sequential(
            nn.Dropout3d(0.5),
            nn.Linear(512 * 7 * 7 * 7, 4096),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout3d(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Dropout3d(0.5),
            nn.Linear(4096, latent_dim),
        )
        self.fc_var = nn.Sequential(
            nn.Dropout3d(0.5),
            nn.Linear(4096, latent_dim),
        )

    def forward(self, input_batch):
        out = self.block1(input_batch)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        return self.fc_mu(out), self.fc_var(out)
    
class FullDiskDecoder(nn.Module):
    def __init__(self, out_channels, latent_dim, bias=False) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        # Shape required to start transpose convs (hid_channels, kernel_size, kernel_size)
        self.reshape = (512, 7, 7, 7)
        self.fc = nn.Linear(latent_dim, 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 512 * 7 * 7 * 7)

        self.block1 = FullDiskTransposeBlock3C(512, conv_channels=512)
        self.block2 = FullDiskTransposeBlock3C(512, conv_channels=512)
        self.block3 = FullDiskTransposeBlock3C(512, conv_channels=256)
        self.block4 = FullDiskTransposeBlock2C(256, conv_channels=128)
        self.block5 = FullDiskTransposeBlock2C(128, conv_channels=64)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_batch):
        batch_size = input_batch.size(0)

        # Fully connected layers with ReLu activations
        out = F.relu(self.fc(input_batch))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
       
        return self.sigmoid(out) # Output between 0 and 1

    
class FullDiskBlock2C(nn.Module):
    def __init__(self, in_channels, conv_channels, bias=False) -> None:
        super().__init__()
        padding='same'

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_batch):
        out = F.relu(self.conv1(input_batch))
        out = F.relu(self.conv2(out))
        return self.maxpool(out)
    
class FullDiskBlock3C(nn.Module):
    def __init__(self, in_channels, conv_channels, bias=False) -> None:
        super().__init__()
        padding='same'

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.conv3 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_batch):
        out = F.relu(self.conv1(input_batch))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        return self.maxpool(out)
    
class FullDiskTransposeBlock2C(nn.Module):
    def __init__(self, in_channels, conv_channels, bias=False) -> None:
        super().__init__()
        padding=1

        self.conv1 = nn.ConvTranspose3d(in_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.conv2 = nn.ConvTranspose3d(conv_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, input_batch):
        out = F.relu(self.conv1(input_batch))
        out = F.relu(self.conv2(out))
        return self.upsample(out)
    
class FullDiskTransposeBlock3C(nn.Module):
    def __init__(self, in_channels, conv_channels, bias=False) -> None:
        super().__init__()
        padding=1

        self.conv1 = nn.ConvTranspose3d(in_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.conv2 = nn.ConvTranspose3d(conv_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.conv3 = nn.ConvTranspose3d(conv_channels, conv_channels, kernel_size=3, padding=padding, bias=bias)
        self.upsample = nn.Upsample(scale_factor=2)


if __name__ == '__main__':
    # Test model
    model = FullDiskModel()
    input_batch = torch.randn(1, 1, 60, 512, 512) # (batch_size, channels, height, width, depth)
    out, mu, log_var = model(input_batch)
    print(out.shape, mu.shape, log_var.shape)