from environment import *


class Agent:
    """
    An agent class.

    Representing single agent which can be a part of several environments.
    """

    def __init__(self, env: MazeEnvironment, state: tuple[int, int], actions: list[Action]):
        self.env = env
        self.state: tuple[int, int] = state
        self.actions = actions

    def get_actions(self):
        return self.get_actions()

    def take_action(self, a: Action):
        if a not in self.actions:
            raise Exception(f'Agent itself cannot take action {a.name}')

        next_s, r, _ = self.env(self.state, a)


if __name__ == '__main__':
    print('Hi! Here you can find implementation of Agent class, the one who learns.')
