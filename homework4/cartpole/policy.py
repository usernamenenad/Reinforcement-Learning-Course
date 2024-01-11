from abc import ABC, abstractmethod
from random import random, choice

from cartpole.utils import controls, Q


class Policy(ABC):
    @abstractmethod
    def act(self, q: Q, state: float) -> float:
        pass


class GreedyPolicy(Policy):
    def act(self, q: Q, state: float) -> float:
        return max([(a, q[state, a]) for a in controls], key=lambda x: x[1])[0]


class RandomPolicy(Policy):
    def act(self, q: Q, state: float) -> float:
        return choice(controls)


class EpsGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def act(self, q: Q, state: float) -> float:
        return GreedyPolicy().act(q, state) if random() > self.epsilon else RandomPolicy().act(q, state)
