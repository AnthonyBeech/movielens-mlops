import hydra
from omegaconf import DictConfig, OmegaConf

from movielens.config.config import PROJECT_ROOT
from movielens.train import run_training
from movielens.utils.logger import setup_logging

setup_logging(__name__)


@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="config")
def main(cfg: DictConfig) -> None:
    """Train models."""
    print("Starting training with configuration:")
    print(OmegaConf.to_yaml(cfg=cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
