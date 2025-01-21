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


class ClassicFeature(BaseFeature):
    def __init__(self, cfg: DictConfig, ccfg: DataColumnsConfig) -> None:
        self.cfg = cfg
        self.ccfg = ccfg
        self.df = pd.DataFrame

    @task()
    def load(self) -> None:
        self.df = load_data(self.cfg.data.ratings_raw, n=self.cfg.exp.n_rows)

    @task()
    def clean(self) -> None:
        self.df = remove_nulls(
            self.df, subset=[self.ccfg.movie_id, self.ccfg.rating, self.ccfg.timestamp, self.ccfg.user_id]
        )
        self.df = keep_by_count(self.df, self.ccfg.movie_id, min_count=self.cfg.exp.min_movie_rating_count)

    @task()
    def transform(self) -> None:
        self.x = self.df[[ccfg.user_id, ccfg.movie_id]]
        self.y = self.df[ccfg.rating]

    @task()
    def validate(self) -> None:
        try:
            self.df = ratings_schema.validate(self.df)
        except pa.errors.SchemaError:
            msg = "Schema fail."
            log.exception(msg)
            raise

    @task()
    def write(self) -> None:
        write_data(self.df, path=self.cfg.data.ratings_processed)

    @flow()
    def run(self) -> None:
        self.load()
        log.info(f"df size: {len(self.df)}")
        self.clean()
        self.validate()
        self.transform()
        log.info(f"self.df size: {len(self.df)}")
        self.write()
