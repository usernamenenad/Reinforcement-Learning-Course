from abc import ABC, abstractmethod
from random import choice

from bandit.utils import *


class Policy(ABC):
    @abstractmethod
    def act(self, q: Q) -> Bandit:
        pass


class GreedyPolicy(Policy):
    def act(self, q: Q) -> Bandit:
        return max([(bandit, q[bandit]) for bandit in q], key=lambda x: x[1])[0]


class RandomPolicy(Policy):
    def act(self, q: Q) -> Bandit:
        return choice([bandit for bandit in q])


class EpsGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.1) -> None:
        self.__epsilon = epsilon

    def act(self, q: Q) -> Bandit:
        return (
            RandomPolicy().act(q)
            if random() < self.__epsilon
            else GreedyPolicy().act(q)
        )
