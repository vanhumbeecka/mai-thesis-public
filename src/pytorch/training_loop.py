from typing import Optional
from common.utils import init_logger
from datetime import timedelta, datetime
import torch

from pytorch.callbacks.latent_kl_loss_writer import LatentKLLossWriter
from pytorch.callbacks.log_image import ImageStoreCallback
from pytorch.dataloader.dl import get_dataloader, get_train_image_set
from pytorch.config import SolarConfig
import lightning as L
from pytorch.models.model_d6_layer4_1_batchnorm import ModelDepth6Layer4Batchnorm
from pytorch.models.model_d6_layer4_1_dropout import (
    ModelDepth6Layer4Dropout,
    load_from_version,
    load_from_version_70,
)
from pytorch.models.model_d6_layer4_2 import ModelDepth6Layer4V2
from pytorch.models.model_d6 import ModelDepth6
from pytorch.models.model_d6_layer4_1 import ModelDepth6Layer4
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import LearningRateFinder


from pytorch.models.model_d6_v2 import ModelDepth6Version2
from pytorch.trainer import LitAutoEncoder
from pytorch.models.simple import SimpleModel
from lightning.pytorch import loggers as pl_loggers
from pytorch.callbacks.printout import MyPrintingCallback

from pytorch.dataloader.dl import FlareEvent

logger = init_logger(__name__)


def run(
    config: SolarConfig,
    batch_size: int,
    num_workers: int,
    latent_size: int,
    learning_rate: int,
    max_epochs: int,
    dropout: float,
    init_weights: bool,
    weight_decay: float,
    gamma: float,
    beta: float,
    pin_memory: bool,
):
    dataloader, dataset = get_dataloader(
        config, batch_size, num_workers, shuffle=True, pin_memory=pin_memory
    )

    # Get a sample image for logging
    flare_idx = 1
    np_array, label, filenames = dataset[flare_idx]

    ###################
    # MODEL SELECTION #
    ###################

    # model = ModelDepth6Layer4(in_chan=1, latent_size=latent_size)
    # model = ModelDepth6Layer4Batchnorm(in_chan=1, latent_size=latent_size)
    model = ModelDepth6Layer4Dropout(
        in_chan=1, latent_size=latent_size, dropout=dropout
    )

    # model = ModelDepth6Layer4V2(
    #     in_chan=1, latent_size=latent_size, dropout=dropout, init_weights=init_weights
    # )

    # model: ModelDepth6Layer4Dropout = load_from_version_70(
    #     config.pl_path, dropout=dropout
    # )

    # model: ModelDepth6Layer4Dropout = load_from_version(
    #     config.pl_path, dropout=dropout, version=105
    # )

    # model = ModelDepth6Version2(
    #     in_chan=1, latent_size=latent_size, init_weights=init_weights
    # )

    #######################
    # END MODEL SELECTION #
    #######################

    # Create a trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config.data_dir)
    trainer = L.Trainer(
        # limit_train_batches=50, # for debugging
        max_epochs=max_epochs,
        logger=tb_logger,
        default_root_dir=config.data_dir,
        log_every_n_steps=8,
        callbacks=[
            ImageStoreCallback(np_array, every_n_epochs=5),
            LatentKLLossWriter(every_n_batches=200),
            # DeviceStatsMonitor(),
            # LearningRateFinder(
            #     min_lr=1e-06,
            #     max_lr=1e-01,
            #     num_training_steps=100,
            #     mode="exponential",
            #     early_stop_threshold=4.0,
            #     update_attr=True,
            # ),
        ],
    )
    lightning_model = LitAutoEncoder(
        model=model,
        beta=beta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gamma=gamma,
    )
    tb_logger.log_hyperparams(
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "beta": beta,
            "batch_size": batch_size,
            "latent_size": latent_size,
            "learning_rate": learning_rate,
            "model": model.__class__.__name__,
            "model_image_dim": config.image_dimension,
            "max_epochs": max_epochs,
            "dropout": dropout,
            "init_weights": init_weights,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "img_path": config.image_path,
        }
    )

    # Start Training
    torch.set_float32_matmul_precision("high")  # medium or high
    trainer.fit(model=lightning_model, train_dataloaders=dataloader)


def parser():
    import argparse
    import os
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--image-path", type=str, default="../../data/flare_images_preprocessed"
    )
    parser.add_argument("--data-path", type=str, default="../../data")
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--init-weights",
        type=bool,
        default=True,
        help="Initialize weights. Only used for certain model",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="L2 regularization"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Learning rate decay")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for KL loss")
    args = parser.parse_args()

    logger.info(json.dumps(args.__dict__, indent=2, default=str))

    absolute_path = os.path.dirname(__file__)

    data_path = os.path.join(absolute_path, args.data_path)
    image_path = os.path.join(absolute_path, args.image_path)
    config: SolarConfig = SolarConfig(
        local_data_path=data_path,
        image_data_path=image_path,
        image_dimension=(6, 512, 512),
        channels_name=["source_19"],
        timedimension_max_timedelta=timedelta(hours=16),
        flare_data_file="valid_flares_10h.pickle",
    )

    return args, config


if __name__ == "__main__":
    args, config = parser()

    logger.info(f"Using data path: {config.data_path}")
    logger.info(f"Using image path: {config.image_path}")

    # pin_memory=True not working on WSL2. Need to disable for now
    pin_memory = False

    run(
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        latent_size=args.latent_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        dropout=args.dropout,
        init_weights=args.init_weights,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        beta=args.beta,
        pin_memory=pin_memory,
    )
    print("Done.")
