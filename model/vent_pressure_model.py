import torch
import pytorch_lightning as pl
from ranger21 import Ranger21
from model.transformer_lstm import TransformerLstm, SimpleLstm, TransformerOnly


class VentPressureModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerOnly(aux=False)
        self.criterion = torch.nn.L1Loss()
        self.lr = cfg.trainer_config.lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        pressures_pred = self.forward(batch)
        is_inhale = batch["u_outs"] == 0
        loss_org = self.criterion(pressures_pred[is_inhale], batch['pressures'][is_inhale])
        loss = loss_org #+ 0.5 * loss_aux_1 #+ 0.5 * loss_aux_2
        return loss, loss_org #, loss_aux_1, loss_aux_2

    def training_step(self, batch, batch_idx):
        #loss, loss_org, loss_aux_1, loss_aux_2 = self.shared_step(
        loss, loss_org = self.shared_step(batch, batch_idx)
        self.log_dict({'loss_org': loss_org}, #'loss_aux_1': loss_aux_1, 'loss_aux_2': loss_aux_2},
                      prog_bar=True, sync_dist=True, logger=False)
        self.log_dict({'train_loss': loss, 'train_loss_org': loss_org}, #'train_loss_aux_1': loss_aux_1, 'train_loss_aux_2': loss_aux_2},
                      prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # loss, loss_org, loss_aux_1, loss_aux_2 = self.shared_step(
        #     batch, batch_idx)
        loss, loss_org = self.shared_step(batch, batch_idx)
        self.log_dict({'val_loss': loss, 'val_loss_org': loss_org}, #'val_loss_aux_1': loss_aux_1, 'val_loss_aux_2': loss_aux_2},
                      prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # optimizer = Ranger21(self.parameters(), lr=self.lr,
        #                      num_epochs=self.cfg.trainer_config.epoch,
        #                      num_batches_per_epoch=self.cfg.trainer_config.num_batches_per_epoch)
        return optimizer
