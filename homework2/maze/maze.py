from copy import deepcopy
from dataclasses import dataclass
from random import random
from alive_progress import alive_bar
from tabulate import tabulate

import numpy as np

from .base import *


class EnvType(Enum):
    DETERMINISTIC = auto()
    STOCHASTIC = auto()


@dataclass
class Probability:
    """
    Models probabilities as dataclass.
    User has the option to set "deterministic" probablities,
    which means that a certain action has always one and only one direction
    associated with it, or "stochastic" - all random.
    """

    def __init__(
        self,
        base: MazeBase,
        states: list[State],
        actions: list[Action],
        env_type: EnvType,
    ):
        self.__probability: dict[tuple[State, Action], dict[Direction, float]] = dict()
        for s in states:
            for a in actions:
                directions = base.get_directions(s)

                probs = np.round(
                    np.random.dirichlet(np.ones(len(directions)), size=1)[0], 3
                ).tolist()
                self.__probability[(s, a)] = dict()

                match env_type:
                    case EnvType.DETERMINISTIC:
                        probs = [
                            1.0 if direction == ad_map[a] else 0.0
                            for direction in directions
                        ]
                    case EnvType.STOCHASTIC:
                        probs = np.round(
                            np.random.dirichlet(np.ones(len(directions)), size=1)[0], 3
                        ).tolist()

                for i, direction in enumerate(directions):
                    self.__probability[(s, a)][direction] = probs[i]

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
    def base(self) -> MazeBase:
        return self.__base

    @property
    def probabilities(self) -> Probability:
        return self.__probabilities

    def __init__(
        self,
        base: MazeBase,
        states: list[State],
        actions: list[Action],
        env_type: EnvType,
        probabilities: Probability = None,
    ):
        self.__base = base
        self.__states = states
        self.__actions = actions
        self.__probabilities: Probability = (
            probabilities
            if probabilities
            else Probability(base, self.__states, self.__actions, env_type)
        )

        self.__q: dict[tuple[State, Action], float] = {
            (s, a): -10 * random() if not base[s].is_terminal else 0.0
            for s in self.__states
            for a in self.__actions
        }

    def __getitem__(self, key: tuple[State, Action]) -> float:
        return self.__q[key]

    def __setitem__(self, key: tuple[State, Action], value: float):
        self.__q[key] = value

    def __iter__(self):
        return iter(self.__q)

    def __str__(self):
        to_repr = []
        for s, a in self.__q:
            to_repr.append({"State": s, "Action": a, "Value": self.__q[(s, a)]})

        return tabulate(to_repr, headers="keys", tablefmt="rst")


@dataclass
class V:
    """
    Class for representing V values.
    """

    @property
    def states(self) -> list[State]:
        return self.__q.states

    def __init__(self, q: Q):
        self.__q: Q = q
        self.__v: dict[State, float] = {s: self.determine(s) for s in self.__q.states}

    def __getitem__(self, key: State) -> float:
        return self.__v[key]

    def __setitem__(self, key: State, value: float):
        self.__v[key] = value

    def __iter__(self):
        return iter(self.__q)

    def determine(self, s: State) -> float:
        q_sum = list()
        for a in self.__q.actions:
            q_sum.append(
                sum(
                    [
                        self.__q.probabilities[(s, a)][direction] * self.__q[(s, a)]
                        for direction in self.__q.base.get_directions(s)
                    ]
                )
            )
        return max(q_sum)


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
        return self.__q.probabilities

    @property
    def q(self) -> Q:
        return self.__q

    @property
    def v(self) -> V:
        return self.__v

    @property
    def gamma(self) -> float:
        return self.__gamma

    def __init__(
        self,
        base: MazeBase,
        actions: list[Action] = None,
        env_type: EnvType = EnvType.STOCHASTIC,
        gamma: float = 1,
    ):
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

        self.__actions: list[Action] = actions if actions else Action.get_all_actions()

        self.__q = Q(base, self.__states, self.__actions, env_type)
        self.__v = V(self.__q)
        self.__gamma = gamma

    def __call__(self, state: State | Any, action: Action) -> list[dict[str, Any]]:
        """
        Makes possible for environment class to act as a Markov Decision process -
        for a given state and action, it will return new states and rewards.
        """
        next_states = list()

        for direction in self.__base.get_directions(state):
            new_state = self.__base.get_from(state, direction)
            if isinstance(self.__base.nodes[new_state], WallCell):
                new_state = state
            new_cell = self.__base[new_state]

            if isinstance(new_cell, TeleportCell):
                new_state = self.__base.find_position(new_cell.teleport_to)
                new_cell = new_cell.teleport_to

            next_states.append(
                {
                    "direction": direction,
                    "new_state": new_state,
                    "reward": new_cell.reward,
                    "probability": self.probabilities[(state, action)][direction],
                    "is_terminal": new_cell.is_terminal,
                }
            )

        return next_states

    def __update_values(self) -> None:
        """
        Private method for updating Q and V values.
        """
        for s in self.__states:
            if not self.__base[s].is_terminal:
                for a in self.__actions:
                    mdp_ret = self(s, a)
                    # q(s, a) = sum(p(s^+, r | s, a)(r + gamma * max_a^+{q(s^+, a^+)}))
                    self.__q[(s, a)] = sum(
                        [
                            mdp["probability"]
                            * (
                                mdp["reward"]
                                + self.gamma * self.__v.determine(mdp["new_state"])
                            )
                            for mdp in mdp_ret
                        ]
                    )
                self.__v[s] = self.__v.determine(s)

    def compute_values(self, eps: float = 0.01, iterations: int = 1000) -> int:
        """
        Method for converging Q and V values using Bellman's equations.
        """
        with alive_bar(iterations) as bar:
            for k in range(iterations):
                ov = deepcopy(self.__q)
                self.__update_values()
                err = max([abs(self.__q[sa] - ov[sa]) for sa in self.__q])
                if err < eps:
                    return k

                bar()

        return iterations
