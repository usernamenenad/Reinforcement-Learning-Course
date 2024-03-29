from dataclasses import dataclass

from bandit.bandit import *


@dataclass
class Q:
    @property
    def q(self) -> dict[Bandit, float]:
        return self.__q

    def __init__(self, bandits: list[Bandit]) -> None:
        self.__bandits: list[Bandit] = bandits
        self.__q: dict[Bandit, float] = {bandit: 0.0 for bandit in self.__bandits}

    def __getitem__(self, key: Bandit) -> float:
        return self.__q[key]

    def __setitem__(self, key: Bandit, value: float) -> None:
        self.__q[key] = value

    def __iter__(self):
        return iter(self.__q)
