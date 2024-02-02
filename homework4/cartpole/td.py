from abc import ABC, abstractmethod
from copy import deepcopy
from random import uniform

from alive_progress import alive_bar

from cartpole.info import Info
from cartpole.model import Cartpole
from cartpole.policy import Policy
from cartpole.utils import *

x_threshold = 5.0
o_threshold = round(radians(20), round_prec)


class TD(ABC):
    @abstractmethod
    def run(
        self,
        model: Cartpole,
        policy: Policy,
        gamma: float = 1.0,
        alpha: float = 0.1,
        iterations: int = 10000,
        T: float = 0.01,
    ) -> Q:
        pass


class SARSA(TD):
    def __init__(self) -> None:
        self.__ss: State = self.__initialize_ss()
        self.__result: dict[int, bool] = {}

    def __initialize_ss(self) -> State:
        return (
            round(uniform(-x_threshold, x_threshold), round_prec),
            0.0,
            round(uniform(-o_threshold, o_threshold), round_prec),
            0.0,
        )

    def run(
        self,
        model: Cartpole,
        policy: Policy,
        actions: list[Action],
        gamma: float = 1.0,
        alpha: float = 0.1,
        iterations: int = 10000,
        T: float = 0.01,
    ) -> Q:
        self.__q = Q(actions)

        with alive_bar(iterations) as bar:
            for i in range(iterations):
                if i % 100 == 0:
                    self.__ss = self.__initialize_ss()
                s = deepcopy(self.__ss)
                a = policy.act(self.__q, self.__ss)
                new_state = model(self.__ss, a, T)

                if (
                    -x_threshold < new_state[0] < x_threshold
                    and -o_threshold < new_state[2] < o_threshold
                ):
                    new_action = policy.act(self.__q, new_state)
                    q_plus = self.__q[new_state, new_action]
                    r = 10
                    self.__ss = new_state
                    self.__result[i] = True
                else:
                    q_plus = 0.0
                    r = -10
                    self.__ss = self.__initialize_ss()
                    self.__result[i] = False

                self.__q[s, a] = (1 - alpha) * self.__q[s, a] + alpha * (
                    r + gamma * q_plus
                )

                bar()

            Info.log_q_values(self.__q, "sarsa")
            Info.log_optimal_policy(self.__q, "sarsa")
            Info.plot_results(self.__result, "sarsa")
            return self.__q
