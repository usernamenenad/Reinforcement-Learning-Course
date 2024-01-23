from math import pow, sin, cos

from cartpole.utils import *

g: float = 9.81
k: float = 1


class Cartpole:
    def __init__(self, m: float, M: float, L: float) -> None:
        self.m = m
        self.M = M
        self.l = L / 2

    def __call__(self, ss: State, a: Action, T: float) -> State:
        return (
            round(ss[0] + T * ss[1], round_prec),
            round(ss[1] + T * self.__F(ss, a), round_prec),
            round(ss[2] + T * ss[3], round_prec),
            round(ss[3] + T * self.__G(ss, a), round_prec),
        )

    def __F(self, ss: State, f: Action) -> float:
        x = ss[0]
        v = ss[1]
        o = ss[2]
        w = ss[3]

        num = 4 * f - self.m * sin(o) * (3 * g * cos(o) - 4 * self.l * pow(w, 2))
        den = 4 * (self.m + self.M) - 3 * self.m * pow(cos(o), 2)
        return num / den

    def __G(self, ss: State, f: Action) -> float:
        x = ss[0]
        v = ss[1]
        o = ss[2]
        w = ss[3]

        num = (self.m * self.M) * g * sin(o) - cos(o) * (
            f + self.m * self.l * sin(o) * pow(w, 2)
        )
        den = self.l * (4 / 3 * (self.m + self.M) - self.m * pow(cos(o), 2))
        return num / den
