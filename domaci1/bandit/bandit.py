import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Iterable, Callable, List, Dict, Tuple
from abc import ABC, abstractmethod, abstractstaticmethod

sns.set()


class Bandit:
    def __init__(self, mean: float, span: float) -> None:
        """
        Initialize the bandit.

        Regardless of the received action, the bandit will return reward 
        uniformly sampled from segment [`mean` - `span`, `mean` + `span`].

        Args:
            `mean` - Mean (expected) value of the reward.
            `span` - Span of the reward.
        """
        self.mean: float = mean
        self.span: float = span

    def pull_leaver(self) -> float:
        """
        Pull leaver and obtain reward.

        Returns:
            The obtained reward.
        """
        return self.mean + 2 * self.span * (random.random() - 0.5)


class BanditsEnvironment:
    """An environment to multiple bandits."""

    def __init__(self, bandits: Iterable[Bandit], penalty: float = 1000.0, stationary: bool = True) -> None:
        """
        Initialize the environment. 

        Args:
            `bandits` - Bandits to be used within the environment.
            `penalty` - If the external agent attempts to use a bandit not in the list, 
                        i.e. if the chosen action is negative or bigger than the index of
                        the last bandit, the returned rewards will be -`penalty`. 
                        Defaults to 1000.
            `stationary` - Is the environment stationary - does it not change over time?
                           Defaults to True.
        """
        self.bandits: List[Bandit] = list(bandits)
        self.penalty: float = penalty
        self.stationary: bool = stationary

    def take_action(self, a: int) -> float:
        """
        Select bandit `a` and pull its leaver. 
        
        If the selected agent is valid, return the obtained reward.
        Otherwise, return negative penalty.
        """
        if a < 0 or a >= len(self.bandits):
            return -self.penalty

        return self.bandits[a].pull_leaver()

    def change_env_chars(self, change_law: Callable[..., Tuple[float, float]], *args) -> None:
        """
        If `self.stationary` is set to False, it will change the
        characteristics of bandits' environment according to passed
        law of change (could be stochastic or deterministic law).
        """

        if self.stationary:
            return

        for bandit in self.bandits:
            bandit.mean, bandit.span = change_law(*args)


class Policy(ABC):
    """
    An interface for policies
    """

    @staticmethod
    @abstractmethod
    def action(**kwargs) -> int:
        pass


class GreedyPolicy(Policy):
    def action(**kwargs) -> int:
        return np.argmax(kwargs['q'])


class RandomPolicy(Policy):
    def action(**kwargs) -> int:
        return random.randint(0, len(kwargs['q']) - 1)


class EpsGreedyPolicy(Policy):
    def action(**kwargs) -> int:
        if random.random() > kwargs['eps']:
            return GreedyPolicy.action(q=kwargs['q'])

        return RandomPolicy.action(q=kwargs['q'])


class Plot(ABC):
    """
    An interface for plots
    """

    @abstractmethod
    def plot(self, env: BanditsEnvironment,
             CHANGE_AT: List[int] = None,
             old_bandit_mean: List[List[float]] = None) -> None:
        pass


class RewardPlot(Plot):
    """
    Plot Mean/Q and Optimal/Taken actions relationships
    """

    def __init__(self, q: List[float], rewards: List[float]) -> None:
        self.rewards = rewards
        self.q = q

    def plot(self, env: BanditsEnvironment,
             CHANGE_AT: List[int] = None,
             old_bandit_mean: List[List[float]] = None) -> None:
        plt.scatter(range(len(env.bandits)), self.q, marker='.')
        plt.scatter(range(len(env.bandits)), [env.bandits[i].mean for i in range(len(env.bandits))], marker='x')

        plt.show()

        g = np.cumsum(self.rewards)
        max_r = max([b.mean for b in env.bandits])

        plt.figure(figsize=(20, 6))
        plt.plot(g)

        legend = ['Collected rewards']

        if not env.stationary:
            for i, old_mean in enumerate(old_bandit_mean):
                old_max_r = max(old_mean)
                plt.plot(np.cumsum(old_max_r * np.ones(len(g))))
                plt.axvline(x=CHANGE_AT[i], ymin=0.05, ymax=0.95, linestyle='dashed', color='red', label='_nolegend_')

                legend.append(f'Maximum reward before {i + 1}th change')

        plt.plot(np.cumsum(max_r * np.ones(len(g))))
        legend.append('Final maximum reward')

        plt.legend(legend, loc='best')
        plt.show()


class ConvergencePlot(Plot):
    """
    Plotting convergence of Q for a `bandit_id` to its' mean value.
    The largest x-axis will show what bandit was played the most and we expect his mean value to be
    the biggest amongst all.
    """

    def __init__(self, q_evol: List[Dict[int, List[float]]], eps: float, ATTEMPTS_NO: int) -> None:
        self.q_evol = q_evol
        self.eps = eps
        self.ATTEMPTS_NO = ATTEMPTS_NO

    def plot(self, env: BanditsEnvironment,
             CHANGE_AT: List[int] = None,
             old_bandit_mean: List[List[float]] = None) -> None:
        plt.figure(figsize=(20, 12))
        legend = []
        start = 0

        colors = sns.color_palette('husl', n_colors=len(env.bandits))

        for i, bandit in enumerate(env.bandits):
            if CHANGE_AT:
                for j in range(len(CHANGE_AT)):
                    xaxis = [k for k in range(start, CHANGE_AT[j])]
                    for k, old_mean in enumerate(old_bandit_mean[j]):
                        plt.plot(xaxis,
                                 old_mean * np.ones(len(xaxis)),
                                 linestyle='--', label='_nolegend_',
                                 color=colors[k])
                    start = CHANGE_AT[j]
            plt.plot(range(CHANGE_AT[-1] if CHANGE_AT else 0, self.ATTEMPTS_NO),
                     bandit.mean * np.ones(self.ATTEMPTS_NO - CHANGE_AT[-1] if CHANGE_AT else self.ATTEMPTS_NO),
                     linestyle='--', label='_nolegend_',
                     color=colors[i])
            plt.scatter([i for i in range(self.ATTEMPTS_NO + 1)], self.q_evol[i], marker='.', s=10, color=colors[i])
            legend.append(f'Q over time for bandit {i}')

        plt.legend(legend, loc='best')
        plt.show()


class System:
    """
    Here we will implement the whole system (environment). If the 
    environment has variable characteristics, the logic for 
    that will be implemented here.
    """

    def __init__(self, bandits: List[Bandit], stationary: bool = True, ALPHA: float = 0.1):
        self.env = BanditsEnvironment(bandits, stationary=stationary)
        self.ALPHA = ALPHA

    def run_system(self, eps: float = 0, ATTEMPTS_NO: int = 10000, CHANGE_AT: List[int] = None):
        if CHANGE_AT:
            CHANGE_AT = list(CHANGE_AT)
            chiter = 0

        q = [0 for _ in range(len(self.env.bandits))]
        q_evol = {i: [0] for i in range(len(self.env.bandits))}
        rewards = []
        old_bandit_mean = []

        if not self.env.stationary:
            def change_law():
                return 10 * (random.random() - 0.5), 5 * random.random()

        for attempt in range(ATTEMPTS_NO):

            if CHANGE_AT:
                if attempt == CHANGE_AT[chiter] and not self.env.stationary:
                    old_bandit_mean.append([b.mean for b in self.env.bandits])
                    self.env.change_env_chars(change_law)
                    chiter = chiter + 1 if chiter + 1 < len(CHANGE_AT) else 0

            a = EpsGreedyPolicy.action(q=q, eps=eps)
            r = self.env.take_action(a)

            q[a] = q[a] + self.ALPHA * (r - q[a])

            for i in range(len(self.env.bandits)):
                if i == a:
                    q_evol[i].append(q[a])
                else:
                    q_evol[i].append(None)

            rewards.append(r)

        if not self.env.stationary:
            plotter = RewardPlot(rewards=rewards, q=q)
            plotter.plot(env=self.env, CHANGE_AT=CHANGE_AT, old_bandit_mean=old_bandit_mean)
        else:
            plotter = RewardPlot(rewards=rewards, q=q)
            plotter.plot(env=self.env)

        return q, q_evol, old_bandit_mean


if __name__ == '__main__':
    print("Hi! I am a Bandit module.")
