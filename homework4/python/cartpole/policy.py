from abc import ABC, abstractmethod
from random import choice
from cartpole.utils import *


class Policy(ABC):
    @abstractmethod
    def act(self, q: Q, s: State) -> Action:
        pass


class RandomPolicy(Policy):
    def act(self, q: Q, s: State) -> Action:
        return choice(q.actions)


class GreedyPolicy(Policy):
    def act(self, q: Q, s: State) -> Action:
        return max([(a, q[s, a]) for a in q.actions], key=lambda x: x[1])[0]


class EpsGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon

    def act(self, q: Q, s: State) -> Action:
        return (
            RandomPolicy().act(q, s)
            if random() < self.epsilon
            else GreedyPolicy().act(q, s)
        )
