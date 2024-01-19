from abc import ABC, abstractmethod

from maze.env import MazeEnvironment, Q
from maze.utils import *

class Policy(ABC):

    @abstractmethod
    def act(*args) -> Action:
        pass


class GreedyPolicyQ:
    """
    Greedy policy using purely Q values.
    """
    
    def act(self, s: State, env: MazeEnvironment, q: Q, gamma: float) -> Action:
        q_values: list[tuple[float, Action]] = []
        for a in env.actions:
            q_values.append((q[s, a], a))

        return max(q_values, key=lambda x: x[0])[1]


class GreedyPolicyV:
    """
    Greedy policy using purely V values.
    """

    def act(self, s: State, env: MazeEnvironment, v: dict[State, float], gamma: float) -> Action:
        v_values: list[tuple[float, Action]] = []
        for a in env.actions:
            mdp_return = env(s, a)
            v_sum = 0.0
            for mdp in mdp_return:
                p = mdp["probability"]
                r = mdp["reward"]
                ns = mdp["next_state"]
                v_sum += p * (r + gamma * v[ns])
            v_values.append((v_sum, a))

        return max(v_values, key=lambda x: x[0])[1]            
