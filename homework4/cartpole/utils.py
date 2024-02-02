from enum import Enum
from dataclasses import dataclass
from math import radians
from random import random, uniform

from numpy import arange

round_prec = 3


Action = float


class State:

    def __init__(self, x: float, x_dot: float, o: float, o_dot: float) -> None:
        self.x: float = x
        self.x_dot: float = x_dot
        self.o: float = o
        self.o_dot: float = o_dot

    def __getitem__(self, key: int) -> float:
        match key:
            case 0:
                return self.x
            case 1:
                return self.x_dot
            case 2:
                return self.o
            case 3:
                return self.o_dot
            case _:
                raise IndexError

    def __setitem__(self, key: int, value: float) -> None:
        match key:
            case 0:
                self.x = value
            case 1:
                self.x_dot = value
            case 2:
                self.o = value
            case 3:
                self.o_dot = value
            case _:
                raise IndexError

    def __repr__(self) -> str:
        return f"{self.x}, {self.x_dot}, {self.o}, {self.o_dot}"


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
        if key not in self.__q:
            self.__q[key] = random()
            if key[0] not in self.__states:
                self.__states.append(key[0])

        return self.__q[key]

    def __setitem__(self, key: tuple[State, Action], value: float) -> None:
        self.__q[key] = value

    def determine_v(self, s: State) -> Action:
        to_max: list[tuple[Action, float]] = []
        for a in self.__actions:
            if (s, a) in self.__q:
                to_max.append((a, self.__q[s, a]))
        return max(to_max, key=lambda x: x[1])[0]
