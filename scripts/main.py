import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from prefect import flow

from movielens.conf.config import PROJECT_ROOT
from movielens.conf.schema import DataColumnsConfig
from movielens.features.baseline import BaselineFeature
from movielens.training.baseline import BaselineTrainer

log = logging.getLogger(__name__)
conf_path = str(PROJECT_ROOT / "conf")
ccfg = DataColumnsConfig


@hydra.main(version_base=None, config_path=conf_path, config_name="config")
@flow
def main(cfg: DictConfig) -> None:
    """
    Train models with integrated Optuna hyperparameter tuning and
    MLflow logging directed into Hydra's output folder.
    """
    log.info("Starting training with configuration:")
    log.info(OmegaConf.to_yaml(cfg=cfg))

    bsf = BaselineFeature(cfg, ccfg)
    bsf.run()

    blt = BaselineTrainer(cfg)
    blt.run()


if __name__ == "__main__":
    main()
