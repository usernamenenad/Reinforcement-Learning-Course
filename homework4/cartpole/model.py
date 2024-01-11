from math import sin, cos

g = 9.81


class Cartpole:
    def __init__(self, m: float, M: float, L: float):
        self.m = m
        self.M = M
        self.l = L / 2

    def __call__(self, s: list[float], u: float, T: float = 0.1) -> tuple[float, float, float, float]:
        return s[0] + T * s[1], s[1] + T * self.__F(s, u), s[2] + T * s[3], s[3] + T * self.__G(s, u)

    def __F(self, s: list[float], u: float) -> float:
        num = 4 * u - self.m * sin(s[2]) * (3 * g * cos(s[2]) - 4 * self.l * s[3] ** 2)
        den = 4 * (self.m + self.M) - 3 * self.m * cos(s[2]) ** 2
        return num / den

    def __G(self, s: list[float], u: float) -> float:
        num = (self.m + self.M) * g * sin(s[2]) - cos(s[2]) * (u + self.m * self.l * sin(s[2]) * s[3] ** 2)
        den = self.l * (4 / 3 * (self.m + self.M) - self.m * cos(s[2]) ** 2)
        return num / den
