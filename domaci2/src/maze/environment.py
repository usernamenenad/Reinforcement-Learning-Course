from copy import deepcopy
from random import random

import numpy as np

from .base import *


class MazeEnvironment:
    """
    Wrapper for a maze board that behaves like an MDP environment.

    This is a callable object that behaves like a stochastic MDP
    state transition function - given the current state and action,
    it returns the following state and reward.

    In addition, the environment object is capable of enumerating all
    possible states and all possible actions, as well as determining
    if the state is terminal.
    """

    @property
    def base(self) -> MazeBase:
        return self.__base

    @property
    def states(self) -> list[Position]:
        return self.__states

    @property
    def q_values(self) -> Dict[tuple[Position, Action], float]:
        return self.__q_values

    @property
    def v_values(self) -> Dict[Position, float]:
        return self.__v_values

    @property
    def probabilities(self) -> Dict[tuple[Position, Action], Dict[Direction, float]]:
        return self.__probabilities

    @property
    def gamma(self) -> float:
        return self.__gamma

    def __init__(self, base: MazeBase, gamma: float = 1):
        """
        Initializer for the environment by specifying the underlying
        maze base.
        :param base: A base for maze, i.e. *Graph* or *Board*.
        :param gamma: Discount factor.
        """
        self.__base = base
        self.__states: list[Position] = [
            node
            for node in self.base.nodes
            if self.base[node].is_steppable and not isinstance(self.__base[node], TeleportCell)
        ]

        # Setting probabilities -
        self.__probabilities: Dict[tuple[Position,
                                         Action], Dict[Direction, float]] = dict()
        self.__set_probabilities()

        self.__q_values: Dict[tuple[Position, Action], float] = {
            (s, a): -10 * random() if not self.is_terminal(s) else 0
            for s in self.__states
            for a in self.get_actions()
        }

        self.__v_values: Dict[Position, float] = {
            s: self.determine_v(s) for s in self.__states
        }

        self.__gamma = gamma

    def __call__(self, state, action: Action):
        """
        Makes possible for environment class to act as a Markov Decision process -
        for a given state and action, it will return new states and rewards.
        """
        snext = list()

        if not isinstance(state, Position):
            state = self.base(state)

        for direction in self.base.get_directions(state):
            new_state = self.compute_direction(state, direction)
            new_cell = self.__base[new_state]

            if isinstance(new_cell, TeleportCell):
                new_state = new_cell.to_teleport_to.position
                new_cell = new_cell.to_teleport_to

            snext.append(
                {
                    "Direction": direction,
                    "New state": new_state,
                    "Reward": new_cell.reward,
                    "Probability": self.probabilities[(state, action)][direction],
                    "Is terminal": new_cell.is_terminal,
                }
            )

        return snext

    def __set_probabilities(self):
        """
        Private method for initializing random probabilities.
        Iterating through all states and all possible directions,
        then generating probabilities based on number of directions.
        """
        for s in self.__states:
            for a in self.get_actions():
                no_probs = len(self.base.get_directions(s))
                probabilities = np.round(np.random.dirichlet(
                    np.ones(no_probs), size=1)[0], 3).tolist()
                self.__probabilities[(s, a)] = {}
                for i, direction in enumerate(self.base.get_directions(s)):
                    self.__probabilities[(s, a)][direction] = probabilities[i]

    def __update_values(self):
        """
        Private method for updating Q and V values.
        """
        for s in self.states:
            if not self.is_terminal(s):
                for a in self.get_actions():
                    news = self(s, a)
                    # q(s, a) = sum(p(s^+, r | s, a)(r + gamma * q(s^+, a^+)))
                    self.__q_values[(s, a)] = sum(
                        [
                            new["Probability"] * (new["Reward"] + self.gamma *
                                                  self.determine_v(new["New state"]))
                            for new in news
                        ]
                    )
                self.__v_values[s] = self.determine_v(s)

    def compute_direction(self, state: Position, direction: Direction) -> Position:
        """
        Follow a specific direction in this environment.
        If possible, it will use base for computing direction.
        """

        if direction not in self.get_directions():
            raise Exception(
                f"Agent cannot move in direction {
                    direction.name} in this environment!"
            )

        return self.__base.compute_direction(state, direction)

    def compute_values(self, eps: float = 0.01, max_iter: int = 1000):
        """
        Method for converging Q and V values using Bellman's equations.
        """
        for k in range(max_iter):
            ov = deepcopy(self.q_values)
            self.__update_values()
            err = max([abs(self.__q_values[(s, a)] - ov[(s, a)])
                       for s, a in self.__q_values])
            if err < eps:
                return k

        return max_iter

    def determine_v(self, s: Position):
        """
        Method for determining V values using Q values.
        """
        q = list()
        for a in self.get_actions():
            q.append(
                sum(
                    [
                        self.__probabilities[(s, a)][direction] *
                        self.__q_values[(s, a)]
                        for direction in self.base.get_directions(s)
                    ]
                )
            )
        # v = max_a(q)
        return max(q)

    def get_actions(self):
        """
        Returns actions that are possible to take in this
        environment.
        """
        return Action.get_all_actions()

    def get_directions(self):
        """
        Returns directions that are possible to follow in this
        environment.
        """
        return Direction.get_all_directions()

    def is_terminal(self, state: Position):
        """
        Returns if the state is terminal.
        """
        return self.__base[state].is_terminal
