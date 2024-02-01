from abc import ABC, abstractmethod
from random import random

from .utils import *


class Policy(ABC):
    @abstractmethod
    def act(self, q: Q, s: State) -> Action:
        pass


class RandomPolicy(Policy):
    def act(self, q: Q, s: State) -> Action:
        return Action.HOLD if random() > 0.5 else Action.HIT


class GreedyPolicy(Policy):
    def act(self, q: Q, s: State) -> Action:
        return Action.HIT if q[s, Action.HIT] > q[s, Action.HOLD] else Action.HOLD


class EpsGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon

    def act(self, q: Q, s: State) -> Action:
        return (
            GreedyPolicy().act(q, s)
            if random() > self.epsilon
            else RandomPolicy().act(q, s)
        )


class DealerPolicy(Policy):
    def act(self, q: Q, s: State) -> Action:
        return Action.HIT if s.total < 17 else Action.HOLD
