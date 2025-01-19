import logging

from omegaconf import DictConfig, OmegaConf

from movielens.conf.config import PROJECT_ROOT
from movielens.conf.schema import DataColumnsConfig
from movielens.features.baseline import BaselineFeature
from movielens.training.baseline import BaselineTrainer

from .base import BasePipeline

log = logging.getLogger(__name__)
conf_path = str(PROJECT_ROOT / "conf")
ccfg = DataColumnsConfig


class BaselinePipeline(BasePipeline):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        log.info("Starting training with configuration:")
        log.info(OmegaConf.to_yaml(self.cfg))

        bsf = BaselineFeature(self.cfg, ccfg)
        bsf.run()

        blt = BaselineTrainer(self.cfg)
        blt.run()
