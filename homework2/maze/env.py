from dataclasses import dataclass
from random import random, randint
from typing import Any

from numpy import round, ones
from numpy.random import dirichlet
from tabulate import tabulate

from maze.base import MazeBase
from maze.utils import *


@dataclass
class Probability:
    """
    Models probabilities as dataclass.
    User has the option to set "deterministic" probabilities,
    which means that a certain action has always one and only one direction
    associated with it, or "stochastic" - all random.
    """

    def __init__(
        self,
        base: MazeBase,
        states: list[State],
        actions: list[Action],
        env_type: EnvType,
    ) -> None:
        """
        A wrapper around probabilities.
        """

        self.__probability: dict[tuple[State, Action], dict[Direction, float]] = dict()

        for s in states:
            directions = base.get_directions(s)
            probs: dict[Action, dict[Direction, float]] = {}

            match env_type:
                case EnvType.DETERMINISTIC:
                    for a in actions:
                        self.__probability[s, a] = {}

                        for d in directions:
                            self.__probability[s, a][d] = 1.0 if d == ad_map[a] else 0.0

                case EnvType.STOCHASTIC:
                    for a in actions:
                        self.__probability[s, a] = {}

                        if len(directions):
                            gen = round(
                                dirichlet(ones(len(directions)), size=1)[0], 3
                            ).tolist()
                        else:
                            gen = [0.0 for _ in Direction.get_all_directions()]

                        for d in Direction.get_all_directions():
                            if d in directions:
                                self.__probability[s, a][d] = gen.pop(0)
                            else:
                                self.__probability[s, a][d] = 0.0

    def __getitem__(self, key: tuple[State, Action]) -> dict[Direction, float]:
        return self.__probability[key]

    def __iter__(self):
        return iter(self.__probability)


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
    def type(self) -> EnvType:
        return self.__type

    @property
    def states(self) -> list[State]:
        return self.__states

    @property
    def actions(self) -> list[Action]:
        return self.__actions

    @property
    def probabilities(self) -> Probability:
        return self.__probabilities

    def __init__(
        self,
        base: MazeBase,
        env_type: EnvType = EnvType.STOCHASTIC,
    ) -> None:
        """
        Initializer for the environment by specifying the underlying
        maze base.
        """

        self.__base = base
        self.__type = env_type

        self.__states: list[State] = [
            node
            for node in self.base
            if self.base[node].is_steppable
            and not isinstance(self.__base[node], TeleportCell)
        ]

        self.__actions: list[Action] = Action.get_all_actions()

        self.__probabilities = Probability(
            base=self.__base,
            states=self.__states,
            actions=self.__actions,
            env_type=self.__type,
        )

    def __call__(self, state: State, action: Action) -> list[dict[str, Any]]:
        """
        Makes possible for environment class to act as a Markov Decision process -
        for a given state and action, it will return new states and rewards.
        """
        mdp = list()

        for direction in self.__base.get_directions(state):
            next_state = self.__base.get_from(state, direction)

            if isinstance(self.__base.nodes[next_state], WallCell):
                next_state = state
                new_cell = self.__base[next_state]
                reward = new_cell.reward
            elif next_state == state:
                new_cell = self.__base[next_state]
                reward = -11
            else:
                new_cell = self.__base[next_state]
                reward = new_cell.reward

            if isinstance(new_cell, TeleportCell):
                next_state = self.__base.find_position(new_cell.teleport_to)
                new_cell = new_cell.teleport_to

            mdp.append(
                {
                    "direction": direction,
                    "next_state": next_state,
                    "reward": reward,
                    "probability": self.__probabilities[state, action][direction],
                    "is_terminal": new_cell.is_terminal,
                }
            )

        return mdp

    def is_terminal(self, s: State) -> bool:
        return self.__base[s].is_terminal
