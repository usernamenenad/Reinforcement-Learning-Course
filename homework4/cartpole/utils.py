from enum import Enum
from dataclasses import dataclass
from math import radians
from random import random, uniform

from numpy import arange

round_prec = 3

State = tuple[float, float, float, float]
Action = float

# actions: list[Action] = arange(-0.1, 0.1, 0.05).round(1).tolist()
# actions: list[Action] = [-1.0, 0.0, 1.0]


@dataclass
class Q:
    @property
    def states(self) -> list[State]:
        return self.__states

    @property
    def actions(self) -> list[Action]:
        return self.__actions

    def __init__(self, actions: list[Action]) -> None:
        self.__q: dict[tuple[State, Action], float] = {}
        self.__states: list[State] = []
        self.__actions: list[Action] = actions

    def __iter__(self):
        return iter(self.__q)

    def __getitem__(self, key: tuple[State, Action]) -> float:
        return self.__q.get(key, random())

    def __setitem__(self, key: tuple[State, Action], value: float) -> None:
        self.__q[key] = value

    def add_state(self, state: State) -> None:
        if state not in self.__states:
            self.__states.append(state)

    def determine_v(self, s: State) -> Action:
        to_max: list[tuple[Action, float]] = []
        for a in self.__actions:
            if (s, a) in self.__q:
                to_max.append((a, self.__q[s, a]))
        return max(to_max, key=lambda x: x[1])[0]
