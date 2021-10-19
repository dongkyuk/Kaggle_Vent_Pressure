from datetime import time
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

        target_dict = {self.cfg.data_config.target: torch.tensor(
            breath[self.cfg.data_config.target].values, dtype=torch.float)}
        timestep_dict = {self.cfg.data_config.timestep: torch.tensor(
            breath[self.cfg.data_config.timestep].values, dtype=torch.float)}
        categorical_feature_dict = {feature: torch.tensor(breath[feature].values, dtype=torch.float)
                                    for feature in self.cfg.data_config.categorical_features}
        continuous_feature_dict = {feature: torch.tensor(breath[feature].values, dtype=torch.float)
                                   for feature in self.cfg.data_config.continuous_features}

        return {**target_dict, **timestep_dict, **categorical_feature_dict, **continuous_feature_dict}
