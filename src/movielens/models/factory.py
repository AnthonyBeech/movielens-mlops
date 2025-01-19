from .base import BaseRecommender
from .baseline import BaselineRecommender


class BaseFactory:
    def create(self) -> BaseRecommender:
        raise NotImplementedError


class BaselineRecommenderFactory(BaseFactory):
    def __init__(self) -> None:
        pass

    def create(self) -> BaselineRecommender:
        return BaselineRecommender()


FACTORY_REGISTRY = {
    "baseline": BaselineRecommenderFactory,
}


def get_factory(factory_name: str) -> BaseFactory:
    factory_class = FACTORY_REGISTRY.get(factory_name.lower())
    if not factory_class:
        msg = f"Unknown model type '{factory_name}'."
        raise ValueError(msg)
    return factory_class()
