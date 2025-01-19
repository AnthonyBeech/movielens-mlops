import logging
from abc import ABC, abstractmethod

import pandas as pd

from movielens.models.base import BaseRecommender

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    @property
    @abstractmethod
    def model(self) -> BaseRecommender:
        raise NotImplementedError

    @abstractmethod
    def setup_mlflow(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def train(self, train_df: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_df: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
