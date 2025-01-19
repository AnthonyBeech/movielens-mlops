from dataclasses import dataclass

import pandera as pa
from pandera import Column, DataFrameSchema


@dataclass
class DataColumnsConfig:
    user_id: str = "userId"
    movie_id: str = "movieId"
    rating: str = "rating"
    timestamp: str = "timestamp"


ratings_schema = DataFrameSchema(
    {
        DataColumnsConfig.user_id: Column(int, nullable=False),
        DataColumnsConfig.movie_id: Column(int, nullable=False),
        DataColumnsConfig.rating: Column(float, nullable=False, checks=pa.Check.in_range(0.5, 5.0)),
        DataColumnsConfig.timestamp: Column(int, nullable=False),
    }
)
