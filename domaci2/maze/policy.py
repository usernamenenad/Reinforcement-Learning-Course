from abc import ABC

from actions import *
from environment import MazeEnvironment


class Policy(ABC):
    """
    An interface for policies.
    """
    pass


class GreedyPolicy(Policy):
    def q_policy(self, s: tuple[int, int], env: MazeEnvironment, actions: list[Action]) -> Action:
        qpa = []
        for a in actions:
            qpa.append((env.q_values[(s, a)], a))

        return max(qpa, key=lambda x: x[0])[1]

    def v_policy(self, s: tuple[int, int], env: MazeEnvironment, actions: list[Action]) -> Action:
        vpa = []
        for a in actions:
            s_new, r, _ = env(s, a)
            vpa.append((r + env.gamma * env.v_values[s_new], a))

        return max(vpa, key=lambda x: x[0])[1]


if __name__ == '__main__':
    print('Hi! Here you can find implementation of Policy class, used for implementing policies.')
