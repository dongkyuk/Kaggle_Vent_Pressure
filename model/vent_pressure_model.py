import torch
import pytorch_lightning as pl
from model.transformer_lstm import SimpleLstm, TransformerOnly, TransformerWithMlp


class VentPressureModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerOnly()
        self.criterion = torch.nn.L1Loss()
        self.lr = cfg.trainer_cfg.lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        pressures_pred = self.forward(batch)
        # Expiratory phase is not scored
        is_inhale = batch["u_out"] == 0
        loss = self.criterion(pressures_pred[is_inhale], batch['pressure'][is_inhale])
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
