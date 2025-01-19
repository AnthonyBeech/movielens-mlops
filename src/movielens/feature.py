import logging

import pandas as pd
import pandera as pa
from omegaconf import DictConfig
from prefect import flow, task

from .conf.schema import DataColumnsConfig, ratings_schema
from .utils.dataset import keep_by_count, load_data, remove_nulls, write_data

log = logging.getLogger(__name__)
ccfg = DataColumnsConfig


@task()
def load_ratings(cfg: DictConfig) -> pd.DataFrame:
    df = load_data(cfg.data.ratings_raw, n=cfg.n)
    return df


@task()
def clean_ratings(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    df = remove_nulls(df, subset=[ccfg.movie_id, ccfg.rating, ccfg.timestamp, ccfg.user_id])
    df = keep_by_count(df, ccfg.movie_id, min_count=cfg.min_movie_rating_count)
    return df


@task()
def vaidate_ratings(df: pd.DataFrame) -> None:
    try:
        df = ratings_schema.validate(df)
    except pa.errors.SchemaError:
        log.exception()


@flow()
def feature_pipeline(cfg: DictConfig) -> pd.DataFrame:
    df = load_ratings(cfg)
    df = clean_ratings(cfg, df)
    df = vaidate_ratings(df)
    write_data(df, path=cfg.data.ratings_processed)
    return df
