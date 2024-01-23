from copy import deepcopy
from typing import Callable, Optional

from bandit.bandit import Bandit
from bandit.policy import Policy
from bandit.utils import Q


class BanditEnvironment:
    def __init__(
        self, bandits: list[Bandit], penalty: float = 1000.0, is_stationary: bool = True
    ) -> None:
        self.__bandits: list[Bandit] = bandits
        self.__penalty: float = penalty
        self.__is_stationary: bool = is_stationary
        self.__q: Q = Q(self.__bandits)

    def pull_leaver(self, bandit: Bandit) -> float:
        return bandit.pull_leaver() if bandit in self.__bandits else -self.__penalty

    def change_environment(
        self, change_law: Optional[Callable[..., tuple[float, float]]], *args
    ) -> None:
        if self.__is_stationary:
            return

        for bandit in self.__bandits:
            bandit.mean, bandit.span = change_law(*args)

    def run(
        self,
        policy: Policy,
        iterations: int = 10000,
        changes_at: Optional[list[int]] = None,
        change_law: Optional[Callable[..., tuple[float, float]]] = None,
        alpha: float = 0.1,
    ) -> tuple[dict[Bandit, dict[int, float]], dict[Bandit, list[float]], list[float]]:
        if self.__is_stationary or not change_law:
            changes_at: list[int] = [-1]

        changes_at: list[int] = deepcopy(changes_at)
        change_at = changes_at.pop(0)

        q_evol: dict[Bandit, dict[int, float]] = {
            bandit: {0: self.__q[bandit]} for bandit in self.__bandits
        }

        mean_evol: dict[Bandit, list[float]] = {
            bandit: [bandit.mean] for bandit in self.__bandits
        }

        rewards: list[float] = list()

        for game in range(iterations):
            if game == change_at:
                self.change_environment(change_law)
                for bandit in self.__bandits:
                    mean_evol[bandit].append(bandit.mean)
                change_at = changes_at.pop(0) if changes_at else -1

            bandit: Bandit = policy.act(self.__q)
            reward = bandit.pull_leaver()

            self.__q[bandit] = self.__q[bandit] + alpha * (reward - self.__q[bandit])
            q_evol[bandit][game + 1] = self.__q[bandit]
            rewards.append(reward)

        return q_evol, mean_evol, rewards
