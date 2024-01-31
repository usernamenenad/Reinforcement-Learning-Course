from dataclasses import dataclass
from random import random, choice

from tabulate import tabulate
from numpy import ones
from numpy.random import dirichlet

from maze.base import MazeBase
from maze.env import MazeEnvironment
from maze.utils import *


@dataclass
class Q:
    """
    Class for representing Q values.
    """

    @property
    def states(self) -> list[State]:
        return self.__states

    @property
    def actions(self) -> list[Action]:
        return self.__actions

    @property
    def v_table(self) -> dict[State, float]:
        return {s: self.determine_v(s) for s in self.states}

    @property
    def q_table(self) -> dict[tuple[State, Action], float]:
        return self.__q

    def __init__(self, env: MazeEnvironment) -> None:
        self.__states = env.states
        self.__actions = env.actions

        self.__q: dict[tuple[State, Action], float] = {
            (s, a): -10 * random() if not env.is_terminal(s) else 0.0
            for s in self.__states
            for a in self.__actions
        }

    def __getitem__(self, key: tuple[State, Action]) -> float:
        return self.__q[key]

    def __setitem__(self, key: tuple[State, Action], value: float) -> None:
        self.__q[key] = value

    def __iter__(self):
        return iter(self.__q)

    def __str__(self) -> str:
        to_repr = []
        for s, a in self.__q:
            to_repr.append({"State": s, "Action": a, "Value": self.__q[(s, a)]})

        return tabulate(to_repr, headers="keys", tablefmt="rst")

    def determine_v(self, s: State) -> float:
        return max([self.__q[s, a] for a in self.__actions])


@dataclass
class V:
    @property
    def states(self) -> list[State]:
        return self.__states

    @property
    def v_table(self) -> dict[State, float]:
        return self.__v

    def __init__(self, env: MazeEnvironment) -> None:
        self.__states: list[State] = env.states
        self.__v: dict[State, float] = {
            s: -10 * random() if not env.is_terminal(s) else 0.0 for s in self.__states
        }

    def __getitem__(self, s: State) -> float:
        return self.__v[s]

    def __setitem__(self, s: State, value: float) -> None:
        self.__v[s] = value

    def __iter__(self):
        return iter(self.__v)

    def __str__(self) -> str:
        to_repr = []
        for s in self.__v:
            to_repr.append({"State": s, "Value": self.__v[s]})

        return tabulate(to_repr, headers="keys", tablefmt="rst")
