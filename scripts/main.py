import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from movielens.config.config import PROJECT_ROOT
from movielens.train import run_training

log = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path=str(PROJECT_ROOT / "config"), config_name="config")
def main(cfg: DictConfig) -> None:
    """Train models."""
    log.info("Starting training with configuration:")
    log.info(OmegaConf.to_yaml(cfg=cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
