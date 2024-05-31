from lightning import Callback, LightningModule, Trainer
import torch
import os
from torch.utils.tensorboard.writer import SummaryWriter


class LatentKLLossWriter(Callback):
    def __init__(self, every_n_batches=10):
        super().__init__()
        self.every_n_batches = every_n_batches

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ):
        if batch_idx % self.every_n_batches == 0:
            x, y, filenames = batch

            with torch.no_grad():
                pl_module.eval()
                reconst_imgs, mu, logvar = pl_module(x)
                pl_module.train()

            latent_kl = self.latent_kl_loss(reconst_imgs, x, mu, logvar)
            self.write_loss(trainer, batch_idx, latent_kl)

    def latent_kl_loss(self, recon_x, x, mu, logvar):
        latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)

        # This is a sanity check to make sure the two calculations are the same
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_kl = latent_kl.sum()
        assert total_kl - KLD < 1e-5, "KL Divergence calculation is incorrect"

        return latent_kl

    def write_loss(self, trainer: Trainer, batch_idx, latent_kl):
        writer: SummaryWriter = trainer.logger.experiment
        path = os.path.join(writer.get_logdir(), "latent_kl_loss.txt")

        with open(path, "a") as file:
            # file.write(",".join([str(epoch),str(batch_idx), "kl_loss", str(loss)]) + "\n")
            latent_dim = latent_kl.shape[0]
            for i in range(latent_dim):
                file.write(
                    ",".join(
                        [
                            str(trainer.current_epoch),
                            str(batch_idx),
                            str("kl_loss_" + str(i)),
                            str(latent_kl[i].item()),
                        ]
                    )
                    + "\n"
                )
