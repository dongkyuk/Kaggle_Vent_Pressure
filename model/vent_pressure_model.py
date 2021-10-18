import torch
import pytorch_lightning as pl
from ranger21 import Ranger21
from model.transformer_lstm import TransformerLstm, SimpleLstm, TransformerOnly


class VentPressureModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerOnly()
        self.criterion = torch.nn.L1Loss()
        self.lr = cfg.trainer_config.lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        pressures_pred = self.forward(batch)
        is_inhale = batch["u_outs"] == 0
        loss = self.criterion(
            pressures_pred[is_inhale], batch['pressures'][is_inhale])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({'train_loss': loss},
                      prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({'val_loss': loss},
                      prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        optimizer = Ranger21(self.parameters(), lr=self.lr,
                             num_epochs=self.cfg.trainer_config.epoch,
                             num_batches_per_epoch=self.cfg.trainer_config.num_batches_per_epoch)
        return optimizer
