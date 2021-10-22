import torch
from torch.utils.data import Dataset


class VentPressureDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.breaths = df.groupby('breath_id')
        self.breath_ids = list(self.breaths.groups.keys())

    def __len__(self):
        return len(self.breaths)

    def __getitem__(self, idx):
        breath_id = self.breath_ids[idx]
        breath = self.breaths.get_group(breath_id)

        target = {self.cfg.data_cfg.target: breath[self.cfg.data_cfg.target].values}
        timestep = {self.cfg.data_cfg.timestep: breath[self.cfg.data_cfg.timestep].values}
        categorical_features = {feature: breath[feature].values for feature in self.cfg.data_cfg.categorical_features}
        continuous_features = {feature: breath[feature].values for feature in self.cfg.data_cfg.continuous_features}

        all_features = {**target, **timestep, **categorical_features, **continuous_features}
        all_features = {feature: torch.tensor(all_features[feature], dtype=torch.float32) for feature in all_features}

        return all_features