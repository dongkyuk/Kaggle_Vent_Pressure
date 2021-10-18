import torch
from torch.utils.data import Dataset


class VentPressureDataset(Dataset):
    def __init__(self, df):
        self.breaths = df.groupby('breath_id')
        self.breath_ids = list(self.breaths.groups.keys())

    def __len__(self):
        return len(self.breaths)

    def __getitem__(self, idx):
        breath_id = self.breath_ids[idx]
        breath = self.breaths.get_group(breath_id)
        return dict(
            pressures = torch.tensor(breath['pressure'].values, dtype=torch.float),
            rs = torch.tensor(breath['R'].values, dtype=torch.float),
            cs = torch.tensor(breath['C'].values, dtype=torch.float),
            u_ins = torch.tensor(breath['u_in'].values, dtype=torch.float),
            u_outs = torch.tensor(breath['u_out'].values, dtype=torch.float),
            time_steps = torch.tensor(breath['time_step'].values, dtype=torch.float),
        )


