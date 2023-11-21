from abc import ABC, abstractmethod
from environment import MazeEnvironment


class Policy(ABC):
    """
    An interface for policies.
    """
    @abstractmethod
    def take_policy(self, s: tuple[int, int], env: MazeEnvironment):
        pass


class GreedyPolicy(Policy):
    def take_policy(self, s: tuple[int, int], env: MazeEnvironment):
        vals = []
        for a in env.get_actions():
            s_new, r, _ = env(s, a)
            vals.append((env.determine_v(s_new), s, a))

        return max(vals, key=lambda x: x[0])


if __name__ == '__main__':
    print('Hi! Here you can find implementation of Policy class, used for implementing policies.')
