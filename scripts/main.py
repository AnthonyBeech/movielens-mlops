import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from prefect import flow

from movielens.conf.config import PROJECT_ROOT
from movielens.conf.schema import DataColumnsConfig
from movielens.features.baseline import BaselineFeature
from movielens.train import run_training

log = logging.getLogger(__name__)
conf_path = str(PROJECT_ROOT / "conf")
ccfg = DataColumnsConfig


@hydra.main(version_base=None, config_path=conf_path, config_name="config")
@flow
def main(cfg: DictConfig) -> None:
    """Train models."""
    log.info("Starting training with configuration:")
    log.info(OmegaConf.to_yaml(cfg=cfg))

    bsf = BaselineFeature(cfg, ccfg)
    bsf.run()

    run_training(cfg)


if __name__ == "__main__":
    main()
