from environment import *
from policy import *


class Agent:
    """
    An agent class.

    Representing single agent which can be a part of several environments.
    """

    def __init__(self, env: MazeEnvironment, state: tuple[int, int], actions: list[Action]):
        self.env: MazeEnvironment = env
        self.state: tuple[int, int] = state
        self.actions: list[Action] = actions
        self.policy: Policy = None

    def get_actions(self):
        return self.get_actions()

    def set_policy(self, policy: Policy):
        self.policy = policy

    def take_action(self, a: Action, s: tuple[int, int] = None):
        if a not in self.actions:
            raise Exception(f'Agent itself cannot take action {a.name}')

        if not s:
            s = self.state

        return self.env(s, a)

    def take_policy_using_q(self, s: tuple[int, int]):
        qpa = []
        for a in self.actions:
            qpa.append((self.env.q_values[(s, a)], a))

        return max(qpa, key=lambda x: x[0])[1]

    def take_policy_using_v(self, s: tuple[int, int]) -> Action:
        vpa = []
        for a in self.actions:
            s_new, r, _ = self.env(s, a)
            vpa.append((r + self.env.gamma * self.env.v_values[s_new], a))

        return max(vpa, key=lambda x: x[0])[1]


if __name__ == '__main__':
    print('Hi! Here you can find implementation of Agent class, the one who learns.')
