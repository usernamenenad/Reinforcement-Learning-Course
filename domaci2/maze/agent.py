from policy import *
from environment import *


class Agent:
    """
    An agent class.

    Representing single agent which can be a part of several environments.
    """

    @property
    def env(self) -> MazeEnvironment:
        return self.__env

    @env.setter
    def env(self, env: MazeEnvironment):
        self.__env = env

    @property
    def state(self) -> tuple[int, int]:
        return self.__state

    @state.setter
    def state(self, state: tuple[int, int]):
        self.__state = state

    @property
    def actions(self) -> list[Action]:
        return self.__actions

    @actions.setter
    def actions(self, actions: list[Action]):
        self.__actions = actions

    @property
    def policy(self) -> Policy:
        return self.__policy

    @policy.setter
    def policy(self, policy: Policy):
        self.__policy = policy

    def __init__(self, env: MazeEnvironment, actions: list[Action], state: tuple[int, int] = None):
        self.__env: MazeEnvironment = env
        self.__state: tuple[int, int] = state if state else (0, 0)
        self.__actions: list[Action] = actions
        self.__policy: Policy = None

    def get_actions(self) -> list[Action]:
        return self.get_actions()

    def get_policy(self) -> str:
        return self.policy.__class__.__name__

    def set_policy(self, policy: Policy):
        self.policy = policy

    def take_action(self, a: Action, s: tuple[int, int] = None) -> tuple[tuple[int, int], float, bool]:
        if a not in self.actions:
            raise Exception(f'Agent itself cannot take action {a.name}')

        if not s:
            s = self.state

        return self.env(s, a)

    def take_policy_using_q(self, s: tuple[int, int]) -> Action:
        return self.policy.q_policy(s, self.env, self.actions)

    def take_policy_using_v(self, s: tuple[int, int]) -> Action:
        return self.policy.v_policy(s, self.env, self.actions)


if __name__ == '__main__':
    print('Hi! Here you can find implementation of Agent class, the one who learns.')
