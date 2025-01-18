# src/evaluation.py
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .config.columns import MOVIE_ID, RATING, USER_ID
from .models.base import BaseRecommender

log = logging.getLogger(__name__)


def evaluate_model(model: BaseRecommender, test_df: pd.DataFrame) -> dict:
    """Evaluate the model using RMSE on the test set."""
    log.info("evaluating model")
    preds = []
    truths = []
    for _, row in test_df.iterrows():
        preds.append(model.predict(row[USER_ID], row[MOVIE_ID]))
        truths.append(row[RATING])
    rmse = np.sqrt(mean_squared_error(truths, preds))
    return {"rmse": rmse}
