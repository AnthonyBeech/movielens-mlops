import logging

import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn import linear_model

from .base import BaseRecommender

log = logging.getLogger(__name__)


class BaselineRecommender(BaseRecommender):
    """A simple recommender that always predicts the global average rating."""

    def __init__(self, cfg: DictConfig) -> None:
        """Init."""
        self.global_avg = None
        self.ratings_df = None
        self.cfg = cfg
        self.model = linear_model

    def fit(self, df: pd.DataFrame) -> None:
        """Fit."""
        self.ratings_df = df
        self.global_avg = df["rating"].mean()
        mlflow.sklearn.log_model(self.model, self.cfg.exp.model.name)

    def predict(self, user_id: list[int], item_id: list[int]) -> list[float]:
        """Predict."""
        log.debug(f"{user_id}, {item_id}")
        return [self.global_avg] * len(user_id)

    def recommend(self, user_id: int, n: int = 10) -> list:
        """Recommend top N."""
        log.debug(f"{user_id}")
        top_items = self.ratings_df.groupby("item_id")["rating"].mean().sort_values(ascending=False).head(n)
        return list(top_items.index)
