from .environment import *
from .policy import *


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

    def __init__(
        self, env: MazeEnvironment, actions: list[Action], state: tuple[int, int] = None
    ):
        self.__env: MazeEnvironment = env
        self.__state: tuple[int, int] = state if state else (0, 0)
        self.__actions: list[Action] = actions

    def take_action(self, a: Action, s: tuple[int, int] = None):
        if a not in self.actions:
            raise Exception(f"Agent itself cannot take action {a.name}")

        if not s:
            s = self.state

        return self.env(s, a)

    def determine_optimal_actions(
        self, policy: Policy
    ) -> Dict[tuple[int, int], Action]:
        sa: Dict[tuple[int, int], Action] = {}
        for s in self.env.states:
            if not self.env.is_terminal(s):
                sa[s] = policy.take_policy(s, self.env, self.actions)

        return sa

    def take_policy(self, s: tuple[int, int], policy: Policy) -> Action:
        return policy.take_policy(s, self.env, self.actions)
