import os
from typing import Optional
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class NeptuneConfig:
    use_neptune: Optional[bool] = None
    api_key: str = os.environ["NEPTUNE_API_KEY"]
    project_name: str = "dongkyuk/vent-pressure"
    exp_name: Optional[str] = "only transformer + time2vec concat lstm + constant lr"
    description: str = ""


@dataclass
class TrainerConfig:
    epoch: int = 1000
    lr: float = 1e-5
    n_gpus: int = 2
    num_workers: int = n_gpus * 4
    seed: int = 42
    fold_num: int = 5
    pin_memory: bool = True
    persistent_workers: bool = True
    train_batch_size: int = 80#120 #256
    val_batch_size: int = 80 #120 #256
    num_batches_per_epoch: int = 252


@dataclass
class PathConfig:
    data_dir: str = "/home/dongkyun/Desktop/Other/Kaggle_Vent_Pressure/data"
    train_data_path: str = os.path.join(data_dir, "train.csv")
    test_data_path: str = os.path.join(data_dir, "test.csv")
    save_dir: Optional[str] = os.path.join(data_dir, "save")


@dataclass
class DataConfig:
    target: str = "pressure"
    timestep: str = "time_step"
    categorical_features: list = field(default_factory=list)
    continuous_features: list = field(default_factory=list)
    

@dataclass
class Config:
    neptune_cfg: NeptuneConfig = NeptuneConfig()
    trainer_cfg: TrainerConfig = TrainerConfig()
    path_cfg: PathConfig = PathConfig()
    data_cfg: DataConfig = DataConfig()
    data_cfg.categorical_features += ["R", "C", "u_out"]
    data_cfg.continuous_features += ["u_in"]
    path_cfg.save_dir = os.path.join(path_cfg.save_dir, neptune_cfg.exp_name)
    os.makedirs(path_cfg.save_dir, exist_ok=True)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
