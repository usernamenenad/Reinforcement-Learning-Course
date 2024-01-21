from abc import ABC, abstractmethod

from maze.env import MazeEnvironment
from maze.func import Q, V
from maze.utils import State, Action


class Policy(ABC):
    @abstractmethod
    def act(self, s: State, env: MazeEnvironment, vf: V | Q, gamma: float) -> Action:
        pass


class GreedyPolicy(Policy):
    def act(self, s: State, env: MazeEnvironment, vf: V | Q, gamma: float) -> Action:
        if isinstance(vf, Q):
            return self.__greedy_policy_q(s, env, vf, gamma)
        return self.__greedy_policy_v(s, env, vf, gamma)

    def __greedy_policy_q(
        self, s: State, env: MazeEnvironment, q: Q, gamma: float
    ) -> Action:
        q_values: list[tuple[float, Action]] = []
        for a in env.actions:
            q_values.append((q[s, a], a))

        return max(q_values, key=lambda x: x[0])[1]

    def __greedy_policy_v(
        self, s: State, env: MazeEnvironment, v: V, gamma: float
    ) -> Action:
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
