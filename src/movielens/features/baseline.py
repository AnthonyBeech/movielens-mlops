import logging

import pandas as pd
import pandera as pa
from omegaconf import DictConfig
from prefect import flow, task

from movielens.conf.schema import DataColumnsConfig, ratings_schema
from movielens.utils.dataset import keep_by_count, load_data, remove_nulls, write_data

from .base import BaseFeature

log = logging.getLogger(__name__)
ccfg = DataColumnsConfig


class BaselineFeature(BaseFeature):
    def __init__(self, cfg: DictConfig, ccfg: DataColumnsConfig) -> None:
        self.cfg = cfg
        self.ccfg = ccfg
        self.df = pd.DataFrame

    @task()
    def load(self) -> pd.DataFrame:
        df = load_data(self.cfg.data.ratings_raw, n=self.cfg.exp.n_rows)
        return df

    @task()
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = remove_nulls(df, subset=[self.ccfg.movie_id, self.ccfg.rating, self.ccfg.timestamp, self.ccfg.user_id])
        df = keep_by_count(df, self.ccfg.movie_id, min_count=self.cfg.exp.min_movie_rating_count)
        return df

    @task()
    def transform() -> None:
        pass

    @task()
    def validate(self, df: pd.DataFrame) -> None:
        try:
            df = ratings_schema.validate(df)
        except pa.errors.SchemaError:
            msg = "Schema fail."
            log.exception(msg)
            raise
        return df

    @task()
    def write(self, df: pd.DataFrame) -> None:
        write_data(df, path=self.cfg.data.ratings_processed)

    @flow()
    def run(self) -> None:
        df = self.load()
        df = self.clean(df)
        df = self.validate(df)
        self.write(df)
