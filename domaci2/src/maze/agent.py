from .policy import *


class Agent:
    """
    A class representing an agent.
    """

    @property
    def env(self) -> MazeEnvironment:
        return self.__env

    @property
    def actions(self) -> list[Action]:
        return self.__actions

    def __init__(self, env: MazeEnvironment, actions: list[Action]):
        """
        :param env: Agent's environment.
        :param actions: Possible actions
                        that agent is able to take (not to be mixed with *environment's possible actions*).
        """
        self.__env: MazeEnvironment = env
        self.__actions: list[Action] = actions

    def take_action(self, state: Position, action: Action):
        if action not in self.actions:
            raise Exception(f"Agent itself cannot take action {action.name}")

        return self.env(state, action)

    def determine_optimal_actions(self, policy: Policy) -> Dict[Position, Action]:
        sa: Dict[Position, Action] = {}
        for s in self.env.states:
            if not self.env.is_terminal(s):
                sa[s] = policy.act(s, self.env, self.actions)

        return sa
