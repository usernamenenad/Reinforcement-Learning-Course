from abc import ABC, abstractmethod
from random import choice
from .utils import *


class Policy(ABC):
    @abstractmethod
    def act(self, q: Q):
        pass


class GreedyPolicy(Policy):
    def act(self, q: Q) -> Bandit:
        return max(q.q, key=q.q.get)


class RandomPolicy(Policy):
    def act(self, q: Q) -> Bandit:
        return choice([bandit for bandit in q])


class EpsGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.1):
        self.__epsilon = epsilon

    def act(self, q: Q) -> Bandit:
        return (
            GreedyPolicy().act(q)
            if random() > self.__epsilon
            else RandomPolicy().act(q)
        )
