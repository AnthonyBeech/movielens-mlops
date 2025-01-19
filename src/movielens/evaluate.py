# src/evaluation.py
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .conf.schema import DataColumnsConfig
from .models.base import BaseRecommender

log = logging.getLogger(__name__)
cfg = DataColumnsConfig


def evaluate_model(model: BaseRecommender, df: pd.DataFrame) -> dict:
    """Evaluate the model using RMSE on the test set."""
    log.info("evaluating model")
    preds = model.predict(df[cfg.user_id], df[cfg.movie_id])
    truths = df[cfg.rating]
    rmse = np.sqrt(mean_squared_error(truths, preds))
    return {"rmse": rmse}
