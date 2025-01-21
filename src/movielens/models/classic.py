import logging

import numpy as np
from sklearn import linear_model

log = logging.getLogger(__name__)


class SKLearnRegression:
    def __init__(self) -> None:
        self.model = linear_model.LinearRegression()


    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> list:
        return self.model.predict(x)

    def recommend(self, user_id: id, n: int = 10) -> list:
        return [user_id] * n
