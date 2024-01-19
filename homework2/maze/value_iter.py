from abc import ABC, abstractmethod
from copy import deepcopy

from alive_progress import alive_bar

from maze import State
from maze.env import Q, MazeEnvironment


class ValueIteration(ABC):

    @abstractmethod
    def run(self, eps: float = 0.1, iterations: int = 1000) -> int:
        pass


class QIteration(ValueIteration):

    def __init__(self, env: MazeEnvironment, gamma: float = 1.0) -> None:
        
        self.env = env
        self.q = Q(base=env.base, states=env.states, actions=env.actions)
        self.gamma = gamma

    def __update_values(self):
        for s in self.env.states:
            if not self.env.base[s].is_terminal:
                for a in self.env.actions:

                    mdp_return = self.env(s, a)
                    
                    # We have to determine the whole Q estimate by using the formula
                    # q(s, a) = sum(p(s^+, r | s, a) * (r + gamma * max_{a^+}{q(s^+, a^+)}))
                    q_sum = 0.0
                    for mdp in mdp_return:
                        p = mdp["probability"]
                        r = mdp["reward"]
                        ns = mdp["next_state"]
                        q_sum += p * (r + self.gamma * self.q.determine_v(ns))
                    self.q[s, a] = q_sum
                    
    def run(self, eps: float = 0.1, iterations: int = 1000):
        with alive_bar(iterations) as bar:
            for iteration in range(iterations):
                oq = deepcopy(self.q)
                self.__update_values()
                err = max(
                    [
                        abs(self.q[s, a] - oq[s, a])
                        for s, a in self.q
                    ]
                )

                if err < eps:
                    return iteration

                bar()

        return iterations


class VIteration(ValueIteration):

    def __init__(self, env: MazeEnvironment, gamma: float = 1.0):
        
        self.env = env
        
        q = Q(base=env.base, states=env.states, actions=env.actions)
        self.v: dict[State, float] = {
            s: q.determine_v(s)
            for s in env.states
        }
        
        self.gamma = gamma

    def __update_values(self):
        for s in self.env.states:
            if not self.env.is_terminal(s):

                # A list of all possible next V values
                v = []

                for a in self.env.actions:

                    # A probability weighted sum
                    v_sum = 0.0

                    mdp_return = self.env(s, a)
                    for mdp in mdp_return:
                        p = mdp["probability"]
                        r = mdp["reward"]
                        ns = mdp["next_state"]
                        v_sum += p * (r + self.gamma * self.v[ns])

                    v.append(v_sum)

                # v(s) = max_{a}{sum(p(s^+, r | s, a) * (r + v(s^+)))}
                self.v[s] = max(v)

    def run(self, eps: float = 0.1, iterations: int = 1000) -> int:
        with alive_bar(iterations) as bar:
            for iteration in range(iterations):
                ov = deepcopy(self.v)
                self.__update_values()
                err = max(
                    [
                        abs(self.v[s] - ov[s])
                        for s in self.v
                    ]
                )

                if err < eps:
                    return iteration

                bar()

        return iterations
