from dataclasses import dataclass

from bandit.bandit import *


@dataclass
class Q:

    @property
    def q(self):
        return self.__q

    def __init__(self, bandits: list[Bandit]):
        self.__bandits: list[Bandit] = bandits
        self.__q: dict[Bandit, float] = {bandit: 0.0 for bandit in self.__bandits}

    def __getitem__(self, key: Bandit):
        return self.__q[key]

    def __setitem__(self, key: Bandit, value: float):
        self.__q[key] = value

    def __iter__(self):
        return iter(self.__q)
