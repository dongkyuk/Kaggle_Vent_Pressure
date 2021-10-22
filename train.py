import os
import hydra
import dotenv
import pytorch_lightning as pl
import glob

from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.plugins import DDPPlugin

dotenv.load_dotenv(override=True)
from model.vent_pressure_model import VentPressureModel
from dataset.datamodule import VentPressureDataModule
from config import register_configs, Config


@hydra.main(config_path=None, config_name="config")
def train(cfg: Config) -> None:
    pl.seed_everything(cfg.trainer_cfg.seed)
    rank_zero_info(OmegaConf.to_yaml(cfg=cfg, resolve=True))

    datamodule = VentPressureDataModule(cfg=cfg, fold=0)
    model = VentPressureModel(cfg=cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.path_cfg.save_dir,
        filename='{epoch:02d}-{val_loss:.4f}.ckpt',
        save_top_k=5,
        mode='min',
    )

    trainer_args = dict(
        gpus=cfg.trainer_cfg.n_gpus,
        accelerator="ddp",
        deterministic=True,
        max_epochs=cfg.trainer_cfg.epoch,
        callbacks=[checkpoint_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
        auto_lr_find=True,
        profiler="simple",
        benchmark=True,
        precision=16,
    )
    if cfg.neptune_cfg.use_neptune:
        logger = NeptuneLogger(
            project_name=cfg.neptune_cfg.project_name,
            experiment_name=cfg.neptune_cfg.exp_name,
            upload_source_files=glob.glob(os.path.join(get_original_cwd(), '**/*.py'), recursive=True),
            api_key=cfg.neptune_cfg.api_key,
            params=cfg.trainer_cfg.__dict__,
        )
        trainer_args['logger'] = logger

    trainer = pl.Trainer(**trainer_args)

    if cfg.trainer_cfg.tune:
        trainer.tune(model, datamodule)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    register_configs()
    train()
