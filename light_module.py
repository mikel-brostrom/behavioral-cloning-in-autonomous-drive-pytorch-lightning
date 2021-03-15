import pytorch_lightning as pl
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from model_architectures.registry import create_model


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model('car_simple_model')

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint["scaler"] = self.scaler

    # def on_load_checkpoint(self, checkpoint):
    #     self.checkpoint = checkpoint["scaler"]

    def training_step(self, batch, _):
        # print('\n\n:FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU', len(batch))
        # print('\n\n:', batch[0].shape, batch[1].shape, flush=True)
        
        # print(x.shape, y.shape)
        # print(x.shape, y.shape)

        x, y = batch
        x, y = x.float(), y.float()
        x = torch.reshape(x, (x.shape[0], x.shape[-1], x.shape[1], x.shape[2]))

        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)

        metrics = {'train/loss': loss, 'step': self.current_epoch}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, _):
        # print('\n\n:FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU', len(batch))
        # print('\n\n:', batch[0].shape, batch[1].shape, flush=True)
        # print('\n\n:', type(batch[0]), type(batch[1]), flush=True)

        x, y = batch
        x, y = x.float(), y.float()
        x = torch.reshape(x, (x.shape[0], x.shape[-1], x.shape[1], x.shape[2]))

        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)

        metrics = {'valid/loss': loss, 'step': self.current_epoch}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        """Initializes a optimizer to use while traning.

        Returns:
            torch.optim.Optimizer: Traning optimizer.
        """
        return torch.optim.Adam(self.model.parameters(), lr=0.02)
