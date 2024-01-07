from dataclasses import dataclass
from random import random
from typing import Callable


class Bandit:
    id = 0

    @property
    def mean(self) -> float:
        return self.__mean

    @mean.setter
    def mean(self, value: float):
        self.__mean = value

    @property
    def span(self) -> float:
        return self.__span

    @span.setter
    def span(self, value: float):
        self.__span = value

    def __init__(self, mean: float = 0.0, span: float = 0.0) -> None:
        self.mean = mean
        self.span = span
        self.__id = Bandit.id
        Bandit.id += 1

    def __hash__(self) -> int:
        return hash(self.__id)

    def __repr__(self) -> str:
        return "Bandit" + str(self.__id)

    def pull_leaver(self) -> float:
        return self.__mean + 2 * self.__span * (random() - 0.5)
