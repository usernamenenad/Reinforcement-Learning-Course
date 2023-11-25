from abc import ABC
from typing import Dict

from .actions import *
from .environment import MazeEnvironment


class Policy(ABC):
    """
    An interface for policies.
    """

    defined_policies: Dict[str, set[str]] = {}


class GreedyPolicy(Policy):
    """
    A class grouping all greedy policies (and their subtypes).
    """

    def __init__(self):
        """
        If we define another greedy policy, it should be
        added into static member `defined_policies` of `Policy` class
        to update list of all possible policies for agent to take.
        """
        if self.__class__.__name__ not in Policy.defined_policies:
            Policy.defined_policies[self.__class__.__name__] = set()
        Policy.defined_policies[self.__class__.__name__].add("greedy_q")
        Policy.defined_policies[self.__class__.__name__].add("greedy_v")

    def q_policy(
            self, s: tuple[int, int], env: MazeEnvironment, actions: list[Action]
    ) -> Action:
        qpa: list[tuple[float, Action]] = []
        for a in actions:
            qpa.append((env.q_values[(s, a)], a))

        return max(qpa, key=lambda x: x[0])[1]

    def v_policy(
            self, s: tuple[int, int], env: MazeEnvironment, actions: list[Action]
    ) -> Action:
        vpa: list[tuple[float, Action]] = []
        for a in actions:
            news = env(s, a)
            v_sum = sum([new['Probability'] * (new['Reward'] + env.gamma *
                        env.v_values[new['New state']]) for new in news])
            vpa.append((v_sum, a))

        return max(vpa, key=lambda x: x[0])[1]


if __name__ == "__main__":
    print(
        "Hi! Here you can find implementation of Policy class, used for implementing policies."
    )
