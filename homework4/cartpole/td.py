from math import radians
from random import uniform
from numpy import arange

from cartpole.model import Cartpole
from cartpole.policy import *
from cartpole.utils import controls, Q


class TD(ABC):
    def __init__(self, s: list[float], q: Q, gamma: float, alpha: float):
        self.s = s
        self.q = q
        self.gamma = gamma
        self.alpha = alpha

    @abstractmethod
    def run(self, cartpole: Cartpole, policy: Policy, iterations: int = 10000):
        pass


class SARSA(TD):

    def __init__(self, s: list[float] = None, q: Q = None, gamma: float = 1.0, alpha: float = 0.1):
        super().__init__(s if s else [uniform(-5.0, 5.0), 0.0, uniform(radians(-90), radians(90)), 0.0],
                         q if q else Q(
                             states=arange(round(radians(-90), 1), round(radians(90) + 0.1, 1), 0.1).round(1).tolist(),
                             actions=controls),
                         gamma,
                         alpha)

    def run(self, cartpole: Cartpole, policy: Policy, iterations: int = 10000, T: float = 0.01) -> Q:

        for _ in range(iterations):

            s = round(self.s[2], 1)
            a = policy.act(self.q, s)
            self.s[0], self.s[1], self.s[2], self.s[3] = cartpole(self.s, a)
            r = 0 if radians(-20) < self.s[2] < radians(20) else -1

            if radians(-20) < self.s[2] < radians(20) and -5.0 < self.s[0] < 5.0:
                ns = round(self.s[2], 1)
                na = policy.act(self.q, ns)
                q_plus = self.q[ns, na]
            else:
                self.s = [uniform(-5.0, 5.0), 0.0, uniform(radians(-90), radians(90)), 0.0]
                q_plus = 0.0

            self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (r + self.gamma * q_plus)

        return self.q
