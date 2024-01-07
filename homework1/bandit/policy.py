from abc import ABC, abstractmethod
import numpy as np
from random import choice, random, randint
from .utils import *


class Policy(ABC):
    @abstractmethod
    def act(self, q: Q):
        pass


class GreedyPolicy(Policy):
    def act(self, q: Q) -> Bandit:
        return max(q, key=q.get)


class RandomPolicy(Policy):
    def act(self, q: Q) -> Bandit:
        return choice([bandit for bandit in q])


class EpsGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.05):
        self.__epsilon = epsilon

    def act(self, q: Q) -> Bandit:
        return (
            RandomPolicy().act(q)
            if random() < self.__epsilon
            else GreedyPolicy().act(q)
        )
