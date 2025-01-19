from abc import ABC, abstractmethod

import pandas as pd


class BaseFeature(ABC):
    """Base class for creating feature set."""

    @abstractmethod
    def load() -> pd.DataFrame:
        """Load the raw data."""
        raise NotImplementedError

    @abstractmethod
    def clean() -> pd.DataFrame:
        """Clean the data."""
        raise NotImplementedError

    @abstractmethod
    def transform() -> pd.DataFrame:
        """Trasnform and create new features in the data."""
        raise NotImplementedError

    @abstractmethod
    def validate() -> pd.DataFrame:
        """Validate the processed data against predifined shema."""
        raise NotImplementedError

    @abstractmethod
    def write() -> None:
        """Write the processed features to a local file."""
        raise NotImplementedError

    @abstractmethod
    def run() -> None:
        """Run the feature processes in sequence"""
        raise NotImplementedError
