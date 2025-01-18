import hydra
from omegaconf import DictConfig, OmegaConf

from movielens.config.config import PROJECT_ROOT
from movielens.train import run_training
from movielens.utils.logger import setup_logging

logger = setup_logging(__name__)


@hydra.main(version_base="1.1", config_path=str(PROJECT_ROOT / "config"), config_name="config")
def main(cfg: DictConfig) -> None:
    """Train models."""
    logger.info("Starting training with configuration:")
    logger.info(OmegaConf.to_yaml(cfg=cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
