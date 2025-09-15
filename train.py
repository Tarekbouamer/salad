import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch

from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule


def _instantiate_loggers(cfg: DictConfig):
    """Handles logger as a single Hydra object or a list of them."""
    if "logger" not in cfg or cfg.logger is None:
        return None
    if isinstance(cfg.logger, list):
        return [hydra.utils.instantiate(lg) for lg in cfg.logger]
    return hydra.utils.instantiate(cfg.logger)


def _instantiate_callbacks(cfg: DictConfig):
    """
    Supports two styles:
    1) Hydra objects list: callbacks: [ {_target_: ...}, ... ]
    2) Dict style like:
       callbacks:
         checkpoint: {...}
         lr_monitor: { enabled: true }
    """
    cbs = []

    # Style 1: list of Hydra-targeted callbacks
    if isinstance(cfg.get("callbacks"), list):
        for cb_conf in cfg.callbacks:
            cbs.append(hydra.utils.instantiate(cb_conf))
        return cbs

    # Style 2: dict with known keys
    if "callbacks" in cfg and cfg.callbacks is not None:
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

        if "checkpoint" in cfg.callbacks and cfg.callbacks.checkpoint is not None:
            cbs.append(ModelCheckpoint(**cfg.callbacks.checkpoint))

        if cfg.callbacks.get("lr_monitor", {}).get("enabled", True):
            cbs.append(LearningRateMonitor(logging_interval="step"))

    return cbs


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # print current path
    print("Current working directory:", hydra.utils.get_original_cwd())

    # Perf hint for RTX 3090 tensor cores
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Show the fully-resolved config once
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # --- Data & Model
    datamodule = GSVCitiesDataModule(
        work_dir=cfg.paths.work_dir, data_dir=cfg.paths.data_dir, **cfg.dataset
    )
    model = VPRModel(**cfg.model)

    # --- Loggers & Callbacks (Hydra-instantiated)
    loggers = _instantiate_loggers(cfg)
    callbacks = _instantiate_callbacks(cfg)

    # --- Trainer
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=loggers)

    # --- Train
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
