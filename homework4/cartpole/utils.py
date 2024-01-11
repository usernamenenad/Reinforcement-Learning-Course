from dataclasses import dataclass
from random import random

controls: list[float] = [-10.0, 10.0]


@dataclass
class Q:

    @property
    def states(self):
        return self.__states

    def __init__(self, states: list[float], actions: list[float]):
        self.__states = states
        self.__actions = actions
        self.__q = {(s, a): random() for s in self.__states for a in self.__actions}

    def __getitem__(self, key: tuple[float, float]) -> float:
        return self.__q[key]

    def __setitem__(self, key: tuple[float, float], value: float) -> None:
        self.__q[key] = value

    def __iter__(self):
        return iter(self.__q)

    def determine_v(self, state: float) -> float:
        return max([self.__q[state, a] for a in self.__actions])
