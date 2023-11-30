from abc import ABC, abstractmethod
from copy import deepcopy
from random import choices, randint
from random import random
from typing import Iterable, Callable, Dict

import numpy as np

from .utils import *
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
    def board(self) -> MazeBoard:
        return self.__board

    @board.setter
    def board(self, board: MazeBoard):
        self.__board = board

    @property
    def states(self) -> list[tuple[int, int]]:
        return self.__states

    @states.setter
    def states(self, states: list[tuple[int, int]]):
        if self.__states:
            self.__states.clear()
        for state in states:
            self.states.append((state[0], state[1]))

    @property
    def q_values(self) -> Dict[tuple[tuple[int, int], Action], float]:
        return self.__q_values

    @q_values.setter
    def q_values(self, q_values: Dict[tuple[tuple[int, int], Action], float]):
        if self.__q_values:
            self.__q_values.clear()
        for state, action in q_values:
            self.__q_values[((state[0], state[1]), action)] = q_values[(state, action)]

    @property
    def v_values(self) -> Dict[tuple[int, int], float]:
        return self.__v_values

    @v_values.setter
    def v_values(self, v_values: Dict[tuple[int, int], float]):
        if self.__v_values:
            self.__v_values.clear()
        for state in v_values:
            self.v_values[(state[0], state[1])] = v_values[state]

    @property
    def probabilities(
            self,
    ) -> Dict[tuple[tuple[int, int], Action], Dict[Direction, float]]:
        return self.__probabilities

    @probabilities.setter
    def probabilities(
            self,
            probabilities: Dict[tuple[tuple[int, int], Action], Dict[Direction, float]],
    ):
        if self.__probabilities:
            self.__probabilities.clear()
        for state, action in probabilities:
            self.__probabilities[((state[0], state[1]), action)] = probabilities[
                (state, action)
            ]

    @property
    def gamma(self) -> float:
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma: float):
        self.__gamma = gamma

    def __init__(self, board: MazeBoard, gamma: float = 1):
        """
        Initializer for the environment by specifying the underlying
        maze board.
        """
        self.__board = board
        self.__states = [
            (i, j)
            for i in range(self.board.rows_no)
            for j in range(self.board.cols_no)
            if self.board[i, j].is_steppable and not isinstance(self.board[i, j], TeleportCell)]

        self.__probabilities: Dict[
            tuple[tuple[int, int], Action], Dict[Direction, float]
        ] = {}

        for s in self.states:
            for a in self.get_actions():
                no_probs = len(self.get_directions())
                probabilities = np.round(
                    np.random.dirichlet(np.ones(no_probs), size=1)[0], 3
                ).tolist()
                self.__probabilities[(s, a)] = {}
                for i, direction in enumerate(self.get_directions()):
                    self.__probabilities[(s, a)][direction] = probabilities[i]

        self.__q_values = {
            (s, a): -10 * random() if not self.is_terminal(s) else 0
            for s in self.states
            for a in self.get_actions()
        }

        self.__v_values = {s: self.determine_v(s) for s in self.states}

        self.__gamma = gamma

    def __call__(self, state: tuple[int, int], action: Action):
        row, col = state
        ss_next = []
        for direction in self.get_directions():
            new_row, new_col = self.compute_direction(row, col, direction)
            new_cell = self.board[new_row, new_col]

            if isinstance(new_cell, TeleportCell):
                new_row = new_cell.to_teleport_to.state[0]
                new_col = new_cell.to_teleport_to.state[1]
                new_cell = new_cell.to_teleport_to

            ss_next.append(
                {
                    "Direction": direction,
                    "New state": (new_row, new_col),
                    "Reward": new_cell.reward,
                    "Probability": self.probabilities[(state, action)][direction],
                    "Is terminal": new_cell.is_terminal,
                }
            )

        return ss_next

    def validate_position(self, row: int, col: int):
        """
        A utility function that validates a position.
        """
        if row < 0 or row >= self.board.rows_no:
            raise Exception("Invalid row position")
        if col < 0 or col >= self.board.cols_no:
            raise Exception("Invalid column position")
        if not self.board[row, col].is_steppable:
            raise Exception("Invalid position: unsteppable cell")

    def compute_direction(self, row: int, col: int, direction: Direction) -> tuple[int, int]:
        """
        Compute a concrete direction for a certain environment.
        Firstly, we define inner functions for movement in all
        4 directions. After, we define the `compute_direction` function itself.
        """

        if direction not in self.get_directions():
            raise Exception(
                f"Agent cannot move in direction {direction.name} in this environment."
            )

        return self.board.connections[(row, col)][direction]

    def determine_v(self, s: tuple[int, int]):
        q = []
        for a in self.get_actions():
            q.append(
                sum(
                    [
                        self.probabilities[(s, a)][direction] * self.q_values[(s, a)]
                        for direction in self.get_directions()
                    ]
                )
            )
        # v = max_a(q)
        return max(q)

    def __update_values(self):
        for s in self.states:
            if not self.is_terminal(s):
                for a in self.get_actions():
                    news = self(s, a)
                    # q(s, a) = sum(p(s^+, r | s, a)(r + gamma * q(s^+, a^+)))
                    self.q_values[(s, a)] = sum(
                        [
                            new["Probability"]
                            * (
                                    new["Reward"]
                                    + self.gamma * self.determine_v(new["New state"])
                            )
                            for new in news
                        ]
                    )
                self.v_values[s] = self.determine_v(s)

    def compute_values(self, eps: float = 0.01, max_iter: int = 1000):
        for k in range(max_iter):
            ov = deepcopy(self.q_values)
            self.__update_values()
            err = max(
                [abs(self.q_values[(s, a)] - ov[(s, a)]) for s, a in self.q_values]
            )
            if err < eps:
                return k

        return max_iter

    def get_actions(self):
        """
        Returns actions that are possible to take in this
        environment.
        """
        return Action.get_all_actions()

    def get_directions(self):
        return Direction.get_all_directions()

    def is_terminal(self, state: tuple[int, int]):
        return self.board[state].is_terminal
