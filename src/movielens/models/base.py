# src/models/base.py
from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Base class for interface of recommender models."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on the provided data."""

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating for a given user-item pair."""

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list:
        """Return top-n recommendations for a given user."""
