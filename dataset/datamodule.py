import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from dataset.dataset import VentPressureDataset


class VentPressureDataModule(pl.LightningDataModule):
    def __init__(self, cfg, fold):
        super(VentPressureDataModule, self).__init__()
        self.cfg = cfg
        self.fold = fold
        self.train_df = pd.read_csv(self.cfg.path_cfg.train_data_path)
        self.test_df = pd.read_csv(self.cfg.path_cfg.test_data_path)
        self._encode_categorical_features()
        self._assign_folds()

    def _encode_categorical_features(self):
        self.encoders = {
            'R': LabelEncoder(),
            'C': LabelEncoder(),
        }
        for col in self.encoders:
            self.train_df[col] = self.encoders[col].fit_transform(self.train_df[col])
            self.test_df[col] = self.encoders[col].transform(self.test_df[col])

    def _assign_folds(self):
        skf = GroupKFold(n_splits=self.cfg.trainer_cfg.fold_num)
        for n, (_, val_index) in enumerate(
            skf.split(
                X=self.train_df,
                y=self.train_df,
                groups=self.train_df['breath_id']
            )
        ):
            self.train_df.loc[val_index, 'kfold'] = int(n)

    def setup(self, stage=None):
        if stage == 'fit':
            self.val_df = self.train_df.loc[self.train_df['kfold'] == self.fold]
            self.train_df = self.train_df.loc[self.train_df['kfold'] != self.fold]
            self.train_dataset = VentPressureDataset(self.cfg, self.train_df)
            self.val_dataset = VentPressureDataset(self.cfg, self.val_df)
        elif stage == 'test':
            self.test_dataset = VentPressureDataset(self.cfg, self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.trainer_cfg.train_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=True, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.trainer_cfg.val_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=False, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.trainer_cfg.val_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=False, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers)
