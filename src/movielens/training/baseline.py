import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from movielens.evaluate import evaluate_model
from movielens.models.base import BaseRecommender
from movielens.models.factory import get_factory
from movielens.utils.dataset import load_data

from .base import BaseTrainer

log = logging.getLogger(__name__)


class BaselineTrainer(BaseTrainer):
    """
    A base trainer that provides common functionality for loading data,
    training a model, evaluating, and logging results with MLflow.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.metrics = {}
        self._model = None

    @property
    def model(self) -> BaseRecommender:
        if self._model is None:
            log.info("Creating model using factory")
            factory = get_factory(self.cfg.exp.model.name)
            self._model = factory.create()
        return self._model

    def setup_mlflow(self) -> None:
        mlflow.set_experiment(self.cfg.exp.mlflow.experiment_name)
        original_cwd = Path(hydra.utils.get_original_cwd())
        tracking_uri = (original_cwd / "mlruns").as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        log.info(f"MLflow tracking URI set to: {tracking_uri}")

    def load(self) -> pd.DataFrame:
        return load_data(self.cfg.data.ratings_processed, n=self.cfg.exp.n_rows)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(df, test_size=self.cfg.training.test_size, random_state=self.cfg.exp.seed)

    def train(self, train_df: pd.DataFrame) -> None:
        log.info("Fitting model")
        self.model.fit(train_df)

    def evaluate(self, test_df: pd.DataFrame) -> None:
        log.info("Evaluating model")
        self.metrics = evaluate_model(self.model, test_df)

    def log_run(self) -> None:
        mlflow.log_param("model_name", self.cfg.exp.model.name)
        mlflow.log_params(self.cfg.exp.model.params)
        mlflow.log_param("test_size", self.cfg.training.test_size)
        mlflow.log_metrics(self.metrics)

    def run(self) -> None:
        log.info("Starting training pipeline")

        self.setup_mlflow()
        df = self.load()
        train_df, test_df = self.split(df)

        with mlflow.start_run():
            self.train(train_df)
            self.evaluate(test_df)
            self.log_run()
