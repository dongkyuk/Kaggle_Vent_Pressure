import os
from typing import Optional
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class NeptuneConfig:
    use_neptune: Optional[bool] = None
    api_key: str = os.environ["NEPTUNE_API_KEY"]
    project_name: str = "dongkyuk/vent-pressure"
    exp_name: Optional[str] = "exp6"
    description: str = "transformer only with categorical embeddings and bigger"


@dataclass
class TrainerConfig:
    epoch: int = 1000
    lr: float = 3e-5
    n_gpus: int = 2
    num_workers: int = n_gpus * 4
    seed: int = 42
    fold_num: int = 5
    pin_memory: bool = True
    persistent_workers: bool = True
    train_batch_size: int = 48
    val_batch_size: int = 48
    num_batches_per_epoch: int = 470


@dataclass
class PathConfig:
    data_dir: str = "/home/dongkyun/Desktop/Other/Kaggle_Vent_Pressure/data"
    train_data_path: str = os.path.join(data_dir, "train.csv")
    test_data_path: str = os.path.join(data_dir, "test.csv")
    save_dir: Optional[str] = os.path.join(data_dir, "save")


@dataclass
class DataConfig:
    target_name: str = "pressure"


@dataclass
class Config:
    neptune_config: NeptuneConfig = NeptuneConfig()
    trainer_config: TrainerConfig = TrainerConfig()
    path_config: PathConfig = PathConfig()
    data_config: DataConfig = DataConfig()

    path_config.save_dir = os.path.join(path_config.save_dir, neptune_config.exp_name)
    os.makedirs(path_config.save_dir, exist_ok=True)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
