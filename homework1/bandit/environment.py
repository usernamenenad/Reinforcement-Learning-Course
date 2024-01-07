from copy import deepcopy
from typing import Callable

from .policy import *


class Environment:
    def __init__(
        self, bandits: list[Bandit], penalty: float = 1000.0, is_stationary: bool = True
    ) -> None:
        self.bandits: list[Bandit] = bandits
        self.penalty: float = penalty
        self.is_stationary: bool = is_stationary
        self.q = Q(self.bandits)

    def pull_leaver(self, bandit: Bandit) -> float:
        return bandit.pull_leaver() if bandit in self.bandits else -self.penalty

    def change_environment(self, change_law: Callable[..., tuple[float, float]], *args):
        if self.is_stationary:
            return

        for bandit in self.bandits:
            bandit.mean, bandit.span = change_law(*args)

    def run(
        self,
        policy: Policy,
        iterations: int = 10000,
        changes_at: list[int] = None,
        change_law: Callable[..., tuple[float, float]] = None,
        alpha: float = 0.1,
    ) -> tuple[dict[Bandit, dict[int, float]], dict[Bandit, list[float]], list[float]]:
        if self.is_stationary or not change_law:
            changes_at = [-1]

        changes_at = deepcopy(changes_at)
        change_at = changes_at.pop(0)
        q_evol: dict[Bandit, dict[int, float]] = {
            bandit: {0: self.q[bandit]} for bandit in self.bandits
        }
        mean_evol: dict[Bandit, list[float]] = {
            bandit: [bandit.mean] for bandit in self.bandits
        }
        rewards: list[float] = list()

        for game in range(iterations):
            if game == change_at:
                self.change_environment(change_law)
                for bandit in self.bandits:
                    mean_evol[bandit].append(bandit.mean)
                change_at = changes_at.pop(0) if changes_at else -1

            bandit = policy.act(self.q)
            reward = bandit.pull_leaver()

            self.q[bandit] = self.q[bandit] + alpha * (reward - self.q[bandit])
            q_evol[bandit][game + 1] = self.q[bandit]
            rewards.append(reward)

        return q_evol, mean_evol, rewards
