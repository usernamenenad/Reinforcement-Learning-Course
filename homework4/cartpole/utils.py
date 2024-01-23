from enum import Enum
from math import radians
from random import random, uniform

from numpy import arange

round_prec = 2

ANGLE_M20 = round(radians(-20), round_prec)
ANGLE_20 = round(radians(20), round_prec)

State = tuple[float, float, float, float]
Action = float

actions: list[Action] = arange(-0.1, 0.1, 0.05).round(1).tolist()


class Q:
    def __init__(self) -> None:
        self.__q: dict[tuple[State, Action], float] = {}
        self.__states: list[State] = []
        self.__actions: list[Action] = actions

    def __iter__(self):
        return iter(self.__q)

    def __getitem__(self, key: tuple[State, Action]) -> float:
        return self.__q.get(key, random())

    def __setitem__(self, key: tuple[State, Action], value: float) -> None:
        self.__q[key] = value

    def add_state(self, state: State):
        if state not in self.__states:
            self.__states.append(state)

    def determine_v(self, s: State) -> Action:
        to_max: list[tuple[Action, float]] = []
        for a in actions:
            if (s, a) in self.__q:
                to_max.append((a, self.__q[s, a]))
        return max(to_max, key=lambda x: x[1])[0]
