from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from typing import Tuple
import os
from mnist.model_burgess import VAE

# https://github.com/pytorch/examples/blob/main/vae/main.py


parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument("--beta", type=float, default=1.0, metavar="B")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
beta_str = "{:.2f}".format(args.beta).replace(".", "_")

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
SUB_FOLDER = os.path.join(DATA_PATH, "mnist-experiments")
print("DATA_PATH: " + DATA_PATH)

train_set = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, **kwargs
)
test_set = datasets.MNIST(DATA_PATH, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, **kwargs
)

# model = VAEMNIST().to(device)
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 32*32), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()
    assert total_kl - KLD < 1e-5

    return BCE + args.beta * KLD, latent_kl


def write_loss(epoch, batch_idx, loss, latent_kl):
    txt_path = os.path.join(SUB_FOLDER, f"latent_kl_loss_beta{beta_str}.txt")
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "a") as file:
        file.write(",".join([str(epoch), str(batch_idx), "kl_loss", str(loss)]) + "\n")
        latent_dim = latent_kl.shape[0]
        for i in range(latent_dim):
            file.write(
                ",".join(
                    [
                        str(epoch),
                        str(batch_idx),
                        str("kl_loss_" + str(i)),
                        str(latent_kl[i].item()),
                    ]
                )
                + "\n"
            )


storer = dict()


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, sample = model(data)
        loss, latent_kl = loss_function(recon_batch, data, mu, logvar)

        latent_dim = latent_kl.shape[0]

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            write_loss(epoch, batch_idx, loss, latent_kl)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_set),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_set)
        )
    )


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, sample = model(data)
            total_loss, latent_kl = loss_function(recon_batch, data, mu, logvar)
            test_loss += total_loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 32, 32)[:n]]
                )
                save_path = os.path.join(SUB_FOLDER, f"reconstruction_{epoch}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(os.path.abspath(save_path))
                save_image(
                    comparison.cpu(),
                    save_path,
                    nrow=n,
                )

    test_loss /= len(test_set)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            # sample = torch.randn(64, 20).to(device)
            sample = torch.randn(64, 10).to(device)
            sample = model.decode(sample).cpu()
            save_path = os.path.join(SUB_FOLDER, "sample_" + str(epoch) + ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(
                sample.view(64, 1, 32, 32),
                save_path,
            )
    name = os.path.join(
        SUB_FOLDER, f"model_burgess_ep{str(args.epochs)}_beta{beta_str}.pt"
    )
    os.makedirs(os.path.dirname(name), exist_ok=True)
    torch.save(model.state_dict(), name)
