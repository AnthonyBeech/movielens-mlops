import hydra
from omegaconf import DictConfig
from prefect import flow

from movielens.conf.config import CONFIG_PATH
from movielens.pipelines.baseline import BaselinePipeline


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
@flow
def main(cfg: DictConfig) -> None:
    pipe = BaselinePipeline(cfg)
    pipe.run()


if __name__ == "__main__":
    main()
