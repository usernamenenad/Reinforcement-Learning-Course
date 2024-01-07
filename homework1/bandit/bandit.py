from random import random


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

    def __init__(self, mean: float, span: float):
        self.mean = mean
        self.span = span
        self.id = Bandit.id
        Bandit.id += 1

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"bandit{self.id}"

    def pull_leaver(self) -> float:
        return self.__mean + 2 * self.__span * (random() - 0.5)
