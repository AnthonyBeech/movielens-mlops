# src/evaluation.py
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from movielens.conf.schema import DataColumnsConfig
from movielens.models.base import BaseRecommender

log = logging.getLogger(__name__)
ccfg = DataColumnsConfig


log = logging.getLogger(__name__)


def evaluate_model(model: BaseRecommender, df: pd.DataFrame) -> dict:
    """
    Evaluate the model on the test set using multiple metrics.

    Metrics calculated:
      - RMSE: Root Mean Squared Error
      - MAE: Mean Absolute Error
      - R2: R-squared (coefficient of determination)

    Parameters:
        model: The trained recommender model.
        df: Test dataset as a DataFrame.
        cfg: Configuration object containing column names (e.g., for user_id, movie_id, rating).

    Returns:
        A dictionary with evaluation metrics.
    """
    log.info("Evaluating model")

    # Predict ratings using the model. Assume cfg contains keys for column names.
    preds = model.predict(df[ccfg.user_id], df[ccfg.movie_id])
    truths = df[ccfg.rating]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(truths, preds))
    mae = mean_absolute_error(truths, preds)
    r2 = r2_score(truths, preds)

    log.info(f"Evaluation metrics: RMSE={rmse}, MAE={mae}, R2={r2}")

    return {"rmse": rmse, "mae": mae, "r2": r2}
