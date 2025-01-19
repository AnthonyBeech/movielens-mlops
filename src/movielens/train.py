import logging

import mlflow
import mlflow.sklearn
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .evaluate import evaluate_model
from .models.factory import get_factory
from .utils.dataset import load_data

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> None:
    """Run training loop."""
    log.info("running training")
    np.random.default_rng(cfg.exp.seed)

    ratings_df = load_data(cfg.data.ratings_processed, n=cfg.exp.n_rows)

    train_df, test_df = train_test_split(ratings_df, test_size=cfg.training.test_size, random_state=cfg.exp.seed)

    factory = get_factory(cfg.exp.model.name)
    model = factory.create()

    log.info("starting experiment")
    mlflow.set_experiment(cfg.exp.mlflow.experiment_name)
    with mlflow.start_run():
        mlflow.log_param("model_name", cfg.exp.model.name)
        mlflow.log_params(cfg.exp.model.params)

        mlflow.log_param("test_size", cfg.training.test_size)

        log.info("fitting model")
        model.fit(train_df)

        metrics = evaluate_model(model, test_df)
        print("Evaluation Metrics:", metrics)
        mlflow.log_metrics(metrics)
