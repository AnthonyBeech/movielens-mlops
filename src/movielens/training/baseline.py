import logging

import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from movielens.models.base import BaseRecommender
from movielens.models.factory import get_factory
from movielens.utils.dataset import load_data
from movielens.utils.evaluate import evaluate_model
from movielens.utils.plotting import Plotter

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
        self.prediction_data = {}
        self._model = None
        self.plotter = Plotter(cfg)

    @property
    def model(self) -> BaseRecommender:
        if self._model is None:
            log.info("Creating model using factory")
            factory = get_factory(self.cfg.exp.model.name)
            self._model = factory.create(self.cfg)
        return self._model

    def setup_mlflow(self) -> None:
        mlflow.set_experiment(self.cfg.exp.mlflow.experiment_name)

    def load(self) -> pd.DataFrame:
        return load_data(self.cfg.data.ratings_processed, n=self.cfg.exp.n_rows)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(df, test_size=self.cfg.training.test_size, random_state=self.cfg.exp.seed)

    def train(self, train_df: pd.DataFrame) -> None:
        log.info("Fitting model")
        self.model.fit(train_df)

    def evaluate(self, test_df: pd.DataFrame) -> None:
        log.info("Evaluating model")
        eval_results = evaluate_model(self.model, test_df)
        self.metrics = eval_results["metrics"]
        self.prediction_data = {
            "preds": eval_results["preds"],
            "truths": eval_results["truths"],
        }

    def log_run(self) -> None:
        mlflow.log_param("model_name", self.cfg.exp.model.name)
        mlflow.log_params(self.cfg.exp.model.params)
        mlflow.log_param("test_size", self.cfg.training.test_size)
        mlflow.log_metrics(self.metrics)
        mlflow.log_param("data_version", self.cfg.data.version)
        mlflow.log_artifact(self.cfg.data.ratings_processed, artifact_path="data")

    def run(self) -> None:
        log.info("Starting training pipeline")

        self.setup_mlflow()
        df = self.load()
        train_df, test_df = self.split(df)

        with mlflow.start_run():
            self.train(train_df)
            self.evaluate(test_df)
            self.log_run()
            self.plotter.log_plots(truths=self.prediction_data["truths"], preds=self.prediction_data["preds"])
