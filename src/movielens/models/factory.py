from omegaconf import DictConfig

from .base import BaseRecommender
from .baseline import BaselineRecommender
from .classic import SKLearnRegression


class BaseFactory:
    def create(self) -> BaseRecommender:
        raise NotImplementedError


class BaselineRecommenderFactory(BaseFactory):
    def __init__(self) -> None:
        pass

    def create(self, cfg: DictConfig) -> BaselineRecommender:
        return BaselineRecommender(cfg)


class SKLearnRegressionFactory(BaseFactory):
    def __init__(self) -> None:
        pass

    def create(self) -> SKLearnRegression:
        return SKLearnRegression()


FACTORY_REGISTRY = {"baseline": BaselineRecommenderFactory, "sklearnregression": SKLearnRegressionFactory}


def get_factory(factory_name: str) -> BaseFactory:
    factory_class = FACTORY_REGISTRY.get(factory_name.lower())
    if not factory_class:
        msg = f"Unknown model type '{factory_name}'."
        raise ValueError(msg)
    return factory_class()
