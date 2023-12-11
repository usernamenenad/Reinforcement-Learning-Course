from typing import Dict

from .environment import MazeEnvironment
from .utils import *


class Policy(ABC):
    """
    An interface for policies.
    """

    defined_policies: Dict[str, set[str]] = {}


class GreedyPolicy(Policy, ABC):
    """
    A class grouping all greedy policies (and their subtypes).
    """

    @abstractmethod
    def take_policy(self, s: Position, env: MazeEnvironment, actions: list[Action]) -> Action:
        pass


class GreedyPolicyQ(GreedyPolicy):
    """
    Inherited from GreedyPolicy - greedy policy using purely Q values.
    """

    def take_policy(self, s: Position, env: MazeEnvironment, actions: list[Action]) -> Action:
        qpa: list[tuple[float, Action]] = list()
        for a in actions:
            qpa.append((env.q_values[(s, a)], a))

        return max(qpa, key=lambda x: x[0])[1]


class GreedyPolicyV(GreedyPolicy):
    """
    Inherited from GreedyPolicy - greedy policy using purely V values.
    """

    def take_policy(self, s: Position, env: MazeEnvironment, actions: list[Action]) -> Action:
        vpa: list[tuple[float, Action]] = list()
        for a in actions:
            news = env(s, a)
            v_sum = sum(
                [
                    new["Probability"] *
                    (new["Reward"] + env.gamma *
                     env.v_values[new["New state"]])
                    for new in news
                ]
            )
            vpa.append((v_sum, a))

        return max(vpa, key=lambda x: x[0])[1]
