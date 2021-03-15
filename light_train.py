import argparse

from light_data import DataModule
from model_architectures.registry import create_model
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from light_module import LightningModel


def train():
    """Train ML model.

    Args:
        config (dict): Configuration.
        experiment (str): Experiment name
        data_dir (str): Folder where to load split from
    """
    model = LightningModel('car_simple_model')
    dm = DataModule()
    dm.prepare_data()
    dm.setup()
    logger = pl.loggers.TensorBoardLogger(save_dir='./')
    checkpoint_callbacks = [
        ModelCheckpoint(monitor="valid/loss", mode="min", filename="best", save_last=True),
    ]
    trainer = pl.Trainer(
        gpus=-1, max_epochs=20, logger=logger, callbacks=checkpoint_callbacks, enable_pl_optimizer=False
    )
    trainer.fit(model, dm)

    logger.finalize("Training success")
    logger.save()


def main(args):
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RUL model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    main(args)
