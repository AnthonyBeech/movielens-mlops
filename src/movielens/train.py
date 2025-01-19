import logging

import mlflow
import mlflow.sklearn
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .evaluate import evaluate_model
from .models.baseline import BaselineRecommender
from .utils.dataset import load_data
from .utils.exception import ModelError

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> None:
    """Run training loop."""
    log.info("running training")
    np.random.default_rng(cfg.seed)

    # Load ratings data
    ratings_df = load_data(cfg.data.ratings_raw, n=cfg.n)

    # Create a train/test split
    train_df, test_df = train_test_split(ratings_df, test_size=cfg.training.test_size, random_state=cfg.seed)

    # Instantiate model; if more models are added later, use an abstract factory or similar pattern.
    if cfg.model.name == "simple":
        model = BaselineRecommender()
    else:
        raise ModelError(cfg.model.name)

    log.info("starting experiment")
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run():
        mlflow.log_param("model_name", cfg.model.name)
        mlflow.log_params(cfg.model.params)

        mlflow.log_param("test_size", cfg.training.test_size)

        log.info("fitting model")
        model.fit(train_df)

        log.info("evaluating model")
        metrics = evaluate_model(model, test_df)
        print("Evaluation Metrics:", metrics)
        mlflow.log_metrics(metrics)
