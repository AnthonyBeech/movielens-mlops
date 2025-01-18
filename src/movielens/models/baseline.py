import pandas as pd

from movielens.utils.logger import setup_logging

from .base import BaseRecommender

logger = setup_logging(__name__)


class BaselineRecommender(BaseRecommender):
    """A simple recommender that always predicts the global average rating."""

    def __init__(self) -> None:
        """Init."""
        self.global_avg = None
        self.ratings_df = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit."""
        self.ratings_df = df
        self.global_avg = df["rating"].mean()

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict."""
        logger.info(f"{user_id}, {item_id}")
        return self.global_avg

    def recommend(self, user_id: int, n: int = 10) -> list:
        """Recommend top N."""
        logger.info(f"{user_id}")
        top_items = self.ratings_df.groupby("item_id")["rating"].mean().sort_values(ascending=False).head(n)
        return list(top_items.index)
