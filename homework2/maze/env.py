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

    def __init__(self, base: MazeBase, states: list[State], actions: list[Action], env_type: EnvType) -> None:
        """
        A wrapper around probabilities.
        """

        self.__probability: dict[tuple[State, Action], dict[Direction, float]] = dict()

        for s in states:
            for a in actions:
                directions = base.get_directions(s)
                self.__probability[s, a] = dict()
                probs = list()

                match env_type:
                    case EnvType.DETERMINISTIC:
                        probs = [1.0 if direction == ad_map[a] else 0.0 for direction in directions]
                        if 1.0 not in probs:
                            if len(probs) < 2:
                                probs = [1.0]
                            else:
                                probs[randint(0, len(probs) - 1)] = 1.0
                    case EnvType.STOCHASTIC:
                        probs = round(dirichlet(ones(len(directions)), size=1)[0], 3).tolist()

                for i, direction in enumerate(directions):
                    self.__probability[s, a][direction] = probs[i]

    def __getitem__(self, key: tuple[State, Action]) -> dict[Direction, float]:
        return self.__probability[key]

    def __iter__(self):
        return iter(self.__probability)


@dataclass
class Q:
    """
    Class for representing Q values.
    """

    @property
    def states(self) -> list[State]:
        return self.__states

    @property
    def actions(self) -> list[Action]:
        return self.__actions

    @property    
    def v_table(self) -> dict[State, float]:
        return {
            s: self.determine_v(s)
            for s in self.states
        }
   
    def __init__(self, base: MazeBase, states: list[State], actions: list[Action]) -> None:
        self.__states = states
        self.__actions = actions

        self.__q: dict[tuple[State, Action], float] = {
            (s, a): -10 * random() if not base[s].is_terminal else 0.0
            for s in self.__states
            for a in self.__actions
        }

    def __getitem__(self, key: tuple[State, Action]) -> float:
        return self.__q[key]

    def __setitem__(self, key: tuple[State, Action], value: float) -> None:
        self.__q[key] = value

    def __iter__(self):
        return iter(self.__q)

    def __str__(self) -> str:
        to_repr = []
        for s, a in self.__q:
            to_repr.append(
                {
                    "State": s,
                    "Action": a,
                    "Value": self.__q[(s, a)]
                }
            )

        return tabulate(to_repr, headers="keys", tablefmt="rst")

    def determine_v(self, s: State) -> float:
        return max([self.__q[s, a] for a in self.__actions])

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

    def __init__(self,
                 base: MazeBase,
                 actions: Optional[list[Action]] = None,
                 env_type: EnvType = EnvType.STOCHASTIC
                 ) -> None:
        """
        Initializer for the environment by specifying the underlying
        maze base.
        """

        self.__base = base
        self.__type = env_type

        self.__states: list[State] = [node
                                      for node in self.base
                                      if self.base[node].is_steppable
                                      and not isinstance(self.__base[node], TeleportCell)]

        self.__actions: list[Action] = actions if actions else Action.get_all_actions()

        self.__probabilities = Probability(base=self.__base,
                                           states=self.__states,
                                           actions=self.__actions,
                                           env_type=self.__type)

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

            if isinstance(new_cell, TeleportCell):
                next_state = self.__base.find_position(new_cell.teleport_to)
                new_cell = new_cell.teleport_to

            mdp.append(
                {
                    "direction": direction,
                    "next_state": next_state,
                    "reward": new_cell.reward,
                    "probability": self.__probabilities[state, action][direction],
                    "is_terminal": new_cell.is_terminal,
                }
            )

        return mdp

    def is_terminal(self, s: State) -> bool:
        return self.__base[s].is_terminal
