import torch
import torch.nn as nn
from torchinfo import summary

# https://poloclub.github.io/cnn-explainer/

class ModelDepth6(nn.Module):
    def __init__(self, in_chan=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_chan, 6, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=0),  # Assuming single-channel input
            nn.ReLU(),
            nn.Conv3d(6, 12, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.Conv3d(12, 24, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(24, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(24, 12, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(12, 6, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(6, in_chan, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=0),
            nn.Sigmoid()
        )


    def forward(self, input_batch):
        batch_size = input_batch.shape[0]
        x = self.encoder(input_batch)

        x = x.view(batch_size, 24, 1, 31, 31)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # Create a random input tensor
    input_channel_size = 1
    input_shape = (6, 512, 512)
    input_tensor = torch.randn(1, input_channel_size, *input_shape)  # Batch size of 1, 3 channels, 32x32 image
    model = ModelDepth6(input_channel_size)
    batch_size = 4

    summary(model, (batch_size, input_channel_size, *input_shape), device='cpu')

    out = model(input_tensor)
    print(input_tensor.shape, out.shape)