from abc import ABC, abstractmethod
from random import random

from .utils import *


class Policy(ABC):
    @abstractmethod
    def act(self, q: Q, state: State) -> Action:
        pass


class RandomPolicy(Policy):
    def act(self, q: Q, state: State) -> Action:
        return Action.HIT if random() > 0.5 else Action.HOLD


class GreedyPolicy(Policy):
    def act(self, q: Q, state: State) -> Action:
        return (
            Action.HIT if q[state, Action.HIT] > q[state, Action.HOLD] else Action.HOLD
        )


class EpsGreedyPolicy(Policy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def act(self, q: Q, state: State) -> Action:
        return (
            GreedyPolicy().act(q, state)
            if random() > self.epsilon
            else RandomPolicy().act(q, state)
        )


class DealerPolicy(Policy):
    def act(self, q: Q, state: State) -> Action:
        return Action.HIT if state.total < 17 else Action.HOLD
