from abc import ABC, abstractmethod


class BasePipeline(ABC):
    @abstractmethod
    def run() -> None:
        raise NotImplementedError
